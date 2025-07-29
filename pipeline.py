import os
import re
import json
import pathlib
import textwrap
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
import sys

import google.generativeai as genai
from dotenv import load_dotenv


MODEL_ID = "gemini-1.5-flash-latest"


class BookProcessingPipeline:
    def __init__(self, 
                 raw_txt_path: pathlib.Path,
                 book_name: str,
                 target_speaker: str,
                 is_narrator: bool,
                 report_path : pathlib.Path = pathlib.Path("runs/usage_report.json")):
        """
        Initializes the full processing pipeline for a book: filtering, splitting, and classifying.
        """

        load_dotenv()
        genai.configure(api_key=os.getenv("GOOGLE_KEY"))
        
        self.model = genai.GenerativeModel(MODEL_ID)

        self.book_name = book_name
        self.book_dir = pathlib.Path("data") / book_name.lower()
        self.book_dir.mkdir(parents=True, exist_ok=True)

        self.raw_txt_path = raw_txt_path
        self.cleaned_txt_path = self.book_dir/"cleaned.txt"

        self.paragraphs_path = self.book_dir / "paragraphs.jsonl"
        self.report_path = pathlib.Path("runs") / book_name.lower() / "usage_report.json"
        self.runs_dir = self.report_path.parent
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        
        self.target_speaker = target_speaker
        self.is_narrator = is_narrator


    def run(self):
        """
        Runs the complete book pipeline:
        - Cleans the raw text by detecting the main content.
        - Splits the text into structured paragraphs.
        - Classifies which paragraphs were spoken or narrated by a target speaker.
        - Saves a usage report.
        """
        print(f"→ Filtering boundaries for {self.book_name} …")
        book_filter_class = BookBoundaryFilter(
            input_path=self.raw_txt_path,
            output_path=self.cleaned_txt_path,
            detected_toc_path=self.book_dir / "detected_toc.txt",
            toc_profile_path=pathlib.Path("utils") / f"struct_profile_{self.book_name.lower()}.json"
        )
        book_filter_class.run()

        print("→ Splitting into structured paragraphs …")
        splitter = StructureSplitter(
            cleaned_txt_path=self.cleaned_txt_path,
            output_path=self.paragraphs_path,
            toc_profile_path=pathlib.Path("utils") / f"struct_profile_{self.book_name.lower()}.json"
        )
        paragraphs = splitter.run(return_paragraphs=True)

        paras = [(p["paragraph_id"], p["text"], p["low_struct_index"]) for p in paragraphs]

        literary_pipeline_class = LiteraryPipeline(
            para_file=self.paragraphs_path,
            runs_dir=self.runs_dir
        )

        print("→ Classifying paragraphs …")
        literary_pipeline_class.classify_book(
            paras=paras,
            target_speaker=self.target_speaker,
            is_narrator=self.is_narrator,
            book_name=self.book_name
        )

        print("✓ Done.")

        self.report_path.write_text(json.dumps(literary_pipeline_class.usage_report, indent=2), encoding="utf-8")
        print(f"✓ Usage report saved to {self.report_path}")



class LiteraryPipeline:
    def __init__(self,
                 para_file: pathlib.Path = pathlib.Path("data/paragraphs.jsonl"),
                 runs_dir: pathlib.Path = pathlib.Path("runs")
                 ):
        """
        Initializes the literary classification pipeline and sets up usage tracking.
        """
        
        load_dotenv()
        genai.configure(api_key=os.getenv("GOOGLE_KEY"))
        self.model = genai.GenerativeModel(MODEL_ID)

        self.para_file = para_file
        self.runs_dir = runs_dir
        self.runs_dir.mkdir(exist_ok=True)

        self.usage_report = {
            "total_prompt_tokens": 0,
            "total_response_tokens": 0,
            "per_run": {}
        }

    def estimate_chunk_size(self, paras, target_words=2000, sample_size=10, min_chunk=5, max_chunk=50):
        """
        Estimates the number of paragraphs per chunk to stay within a target word count.
        """
        word_counts = [len(p[1].split()) for p in paras[:sample_size] if p[1].strip()]
        if not word_counts:
            print("Warning: No valid paragraphs to sample from.")
            return min_chunk
    
        avg_words = sum(word_counts) / len(word_counts)
        est_chunk_size = int(target_words / avg_words)
        bounded_chunk_size = max(min_chunk, min(est_chunk_size, max_chunk))
    
        print(f"=== Avg words per paragraph: {avg_words:.1f}, raw chunk size: {est_chunk_size}, bounded to: {bounded_chunk_size}")
        return bounded_chunk_size
    

    def classify_book(self, paras: List[Tuple[int, str, Dict]], target_speaker: str, is_narrator: bool, book_name: str = "book"):
        """
        Classifies paragraphs by whether the target speaker is speaking or narrating.
        Saves selected paragraphs and logs token usage.
        """
        chunk_sz = min(self.estimate_chunk_size(paras), 50)
        chunks = [paras[i:i+chunk_sz] for i in range(0, len(paras), chunk_sz)]

        keepers = set()
        prompt_total = response_total = 0


        for i, chunk in enumerate(chunks):
            if not chunk:
                continue

            print(f"→ Classifying chunk {i+1}/{len(chunks)}: paragraphs {chunk[0][0]}–{chunk[-1][0]}")
            section_title = f"Chunk of paragraphs {chunk[0][0]}–{chunk[-1][0]}"
            para_block = "\n".join(f"{pid}. {txt}" for pid, txt, _ in chunk)

            narrator_clause = (
                f"- Treat any paragraph outside another character’s quoted speech as narration by {target_speaker}."
                if is_narrator else
                f"- {target_speaker} NEVER narrates. Only label a paragraph if {target_speaker} is explicitly speaking in direct dialogue (e.g., quoted speech or clearly addressed)."
            )

            prompt = textwrap.dedent(f"""
                You are an expert literary analyst.

                TASK:
                For each paragraph below, identify whether {target_speaker} is speaking directly, or narrating.

                CONTEXT:
                - Section title: {section_title}
                {narrator_clause}
                - Characters may be referred to by either full names or common short forms in dialogue.

                FORMAT:
                Return JSON: {{ "keepers": [list of paragraph_ids where {target_speaker} is speaking or narrating] }}

                Paragraphs:
                {para_block}
            """)

            try:
                resp = self.model.generate_content(prompt)
                um = resp.usage_metadata
                prompt_total += um.prompt_token_count
                response_total += um.candidates_token_count

                raw = resp.text.strip()
                obj = json.loads(re.sub(r"```(?:json)?|```", "", raw))
                keepers.update(obj.get("keepers", []))
            except Exception:
                keepers.update(int(n) for n in re.findall(r"\d+", raw))

        out_lines = [json.dumps({
                        "speaker": target_speaker,
                        "paragraph_id": pid,
                        "text": txt,
                        "low_struct_index": lsi
                        }, ensure_ascii=False)
                     for pid, txt, lsi in paras if pid in keepers]

        out_path = self.runs_dir / f"{target_speaker.lower().replace(' ', '')}_segments.jsonl"
        out_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
        print(f"=== Saved {len(out_lines)} paragraphs to {out_path}")

        self.usage_report["per_run"]["classification"] = {"prompt": prompt_total, "response": response_total}
        self.usage_report["total_prompt_tokens"] += prompt_total
        self.usage_report["total_response_tokens"] += response_total


    def run(self, target_speaker: str = "Socrates", is_narrator: bool = True, book_name: str = "book"):
        """
        Runs the classification process using a saved paragraphs file.
        """

        with self.para_file.open(encoding="utf-8") as f:
            paras = [(obj["paragraph_id"], obj["text"], obj["low_struct_index"]) for obj in map(json.loads, f)]

        self.classify_book(paras, target_speaker, is_narrator, book_name)

        report_path = self.runs_dir / "usage_report.json"
        report_path.write_text(json.dumps(self.usage_report, indent=2), encoding="utf-8")
        print(f"✓ Usage report saved to {report_path}")


class BookBoundaryFilter:
    def __init__(
                self, 
                input_path: pathlib.Path, 
                output_path: pathlib.Path,
                structure_profile_path = pathlib.Path("utils/structure_profile.json"),
                detected_toc_path = pathlib.Path("runs/detected_toc.txt"),
                toc_profile_path = pathlib.Path("utils/structure_profile_toc.json")
                ):
        """
        Initializes the book boundary detector with paths and LLM prompts.
        """

        self.input_path = input_path
        self.output_path = output_path
        self.model = genai.GenerativeModel(MODEL_ID)
        
        self.structure_profile_path = structure_profile_path
        self.detected_toc_path = detected_toc_path
        self.toc_profile_path = toc_profile_path

        self.prompt_intro = ("You are a precise literary analyzer. Your task is to inspect the following raw book content "
                            "(which may include prefaces, legal disclaimers, introductions, analyses, etc.) and respond with structured JSON.\n\n"
                            "Your response must be a JSON object with the following keys:\n"
                            "- 'start_type': either 'no_t_o_c' if no Table of Contents is detected, or 't_o_c' if one is clearly present.\n"
                            "- 'search_hint': a short phrase or heading that marks the true start of the main narrative (e.g., 'Chapter I.', 'ACT I.', 'Scene 1.', 'BOOK I.', etc.). This will be used to trim everything before it.\n"
                            "- 't_o_c': if a Table of Contents exists, extract its full text **excluding any generic or non-structural headings** that are not part of the story (such as 'Contents', 'Preface', 'Introduction', 'Introduction and Analysis', 'Analysis', 'About the Author').\n\n"
                            "Additional heuristics:\n"
                            "• TOC entries often appear as 3 or more lines in a row in the very beginning of the book while other lines will be spaced out, centered or spaced out, using capitalized titles like 'The Fall of Days' or 'Chapter I'.\n"
                            "• There may be blank lines between entries.\n"
                            "• A TOC may appear after the publisher block and before an actual chapter or introduction starts.\n\n"
                            "Rules:\n"
                            "• If no TOC is detected, return an empty string for 't_o_c'.\n"
                            "• 'search_hint' must reflect the actual first structural heading found in the main content.\n"
                            "• Your JSON must be machine-readable and minimal.\n\n"
                            "Analyze this content:\n")

        self.prompt_outro = ("You are a precise literary analyzer. Below is the final portion of a book, which may include ending credits, licensing notes, or appendices.\n\n"
                            "Return a JSON object with this key:\n"
                            "- 'search_hint_end': the final sentence or phrase that is part of the *true narrative content* of the book.\n"
                            "Do NOT return disclaimers, license notices, metadata, or postfaces.\n"
                            "Respond ONLY with a minimal JSON object.\n\n"
                            "Analyze this content:\n")

        self.MAX_CHAR_INTRO = 6_000
        self.MAX_CHAR = 3_000

        
    def _call_llm(self, prompt: str, content: str) -> dict:
        """
        Sends a prompt and content to the LLM and parses the JSON response.
        """

        try:
            full = prompt + "\n" + content

            resp = self.model.generate_content(full)

            txt = resp.text.strip()
            if txt.startswith("```"):
                txt = re.sub(r"^```(?:json)?\s*", "", txt)
                txt = re.sub(r"\s*```$", "", txt)
            return json.loads(txt)
        
        except Exception as e:
            print("[LLM ERROR]", e)
            print("Raw response was:\n", resp.text[:300])
            return {}

    def extract_intro_metadata(self, raw_text):
        """
        Extracts metadata from the book's introduction using the LLM.
        """
        return self._call_llm(self.prompt_intro, raw_text[:self.MAX_CHAR_INTRO])

    def extract_outro_metadata(self, raw_text):
        """
        Extracts metadata from the book's ending using the LLM.
        """
        return self._call_llm(self.prompt_outro, raw_text[-self.MAX_CHAR:])
    
    def disambiguate_heading(self, heading: str, raw_text: str) -> int:
        """
        Asks the LLM to disambiguate multiple occurrences of a heading to find the real narrative start.
        """
        
        positions = [m.start() for m in re.finditer(re.escape(heading), raw_text)]
        excerpts = []
        for idx in positions:
            window = raw_text[max(0, idx-400): idx+1000]
            excerpts.append(window)
    
        prompt = (
            f"You are analyzing where a real book starts.\n\n"
            f"The heading **{heading}** appears {len(positions)} times in the text.\n"
            f"I will now show you all of them, and your job is to tell me which one marks "
            f"the actual beginning of the real book (not an analysis, not a mention).\n\n"
            f"Reply only with the number (0-indexed) of the excerpt that starts the book.\n"
        )
        for i, ex in enumerate(excerpts):
            prompt += f"\n## EXCERPT {i}\n{ex[:1200]}\n"
    
        try:
            resp = self.model.generate_content(prompt)
            raw = resp.text.strip()
            m = re.search(r"\d+", raw)
            if m:
                idx = int(m.group(0))
                return positions[idx] if 0 <= idx < len(positions) else 0
        except Exception as e:
            print("[LLM ERROR in disambiguate_heading]", e)
    
        return 0
    
    def first_heading_offset(self, txt: str) -> int:
        """
        Scan the raw text for the first regex match among ALL patterns
        in utils/structure_profile.json. Return byte offset or 0 if none.
        """
        try:
            prof = json.loads(self.structure_profile_path.read_text(encoding="utf-8"))
        except Exception:
            return 0
    
        earliest = len(txt)
        for pat_list in prof.get("examples", {}).values():
            for pat in pat_list:
                try:
                    m = re.search(pat, txt, flags=re.IGNORECASE|re.MULTILINE)
                    if m and m.start() < earliest:
                        earliest = m.start()
                except re.error:
                    continue
        return 0 if earliest == len(txt) else earliest
    
    def run(self):
        """
        Cleans the book text by removing prefaces and appendices,
        detects start and end of the main content, and trims accordingly.
        Optionally processes the table of contents for structure.
        """
        raw = self.input_path.read_text(encoding="utf-8")
        intro_meta = self.extract_intro_metadata(raw)
        outro_meta = self.extract_outro_metadata(raw)

        toc_txt = intro_meta.get("t_o_c", "").strip()
        series_idx = None

        if toc_txt:
            self.toc_profile_path.write_text(toc_txt, encoding="utf-8")
            self.detected_toc_path.write_text(toc_txt, encoding="utf-8")
            print("\n–– DETECTED TABLE OF CONTENTS ––\n")
            print(toc_txt[:3000] + ("…\n" if len(toc_txt) > 1200 else "\n"))

        try:
            SERIES_PAT = re.compile(r'(BOOK|PART|CHAPTER|ACT)\s+([IVXLCDM]+|\d+)\b', re.IGNORECASE)
            matches = list(SERIES_PAT.finditer(raw))
    
            if matches:
                first_heading = matches[0].group(0).strip()
                picked = self.disambiguate_heading(first_heading, raw)
                if picked:
                    series_idx = picked
                    print(f"✓ LLM picked narrative start at byte {series_idx}")
        except ImportError:
            print("[Warn] Missing disambiguation logic — skipping series trimming.")

        try:
            toc_to_profile = TOCToProfile(detected_toc_path=self.detected_toc_path, toc_profile_path=self.toc_profile_path)
            toc_to_profile.build_profile()
        except Exception as e:
            print(f"[Warn] Failed to build TOC profile: {e}")

        if not series_idx and not toc_txt:
            series_idx = self.first_heading_offset(raw)

        if series_idx:
            trimmed = raw[series_idx:]

            if intro_meta.get("start_type") != "t_o_c":
                intro_meta["start_type"] = "no_t_o_c"
            
            intro_meta["search_hint"] = raw[series_idx:series_idx+80]
            intro_meta["t_o_c"] = ""
        else:
            trimmed = self.trim_intro_by_hint(raw, intro_meta.get("search_hint", ""))

        trimmed = self.trim_outro_by_hint(trimmed, outro_meta.get("search_hint_end", ""))

        self.output_path.write_text(trimmed, encoding="utf-8")

        if not toc_txt and self.toc_profile_path.exists():
            self.toc_profile_path.unlink()

            print("No table of contents detected, removed previous table of contents file.")
            print(f"Start type: {intro_meta.get('start_type')} | Start hint: {intro_meta.get('search_hint')[:50]}...")
            print(f"End hint: {outro_meta.get('search_hint_end')[:50]}...")

        print(f"Cleaned text saved to {self.output_path}")
        return {
            **intro_meta, 
            **outro_meta, 
            "detected_toc_path": str(self.detected_toc_path) if toc_txt else None
        }

    @staticmethod
    def trim_intro_by_hint(text: str, hint: str) -> str:
        """
        Trims the text before the first occurrence of the intro hint.
        """
        if not hint:
            return text
        idx = text.find(hint)
        return text[idx:] if idx >= 0 else text

    @staticmethod
    def trim_outro_by_hint(text: str, hint: str) -> str:
        """
        Trims the text after the last occurrence of the outro hint.
        """
        if not hint:
            return text
        idx = text.rfind(hint)
        return text[:idx + len(hint)] if idx >= 0 else text

    @staticmethod
    def find_series_start(text):
        """
        Heuristic to find the second structural heading if the first is often an analysis or preface.
        """
        series_pat = re.compile(r'(BOOK|PART|CHAPTER|ACT)\s+([IVXLCDM]+|\d+)\b', re.IGNORECASE)
        matches = list(series_pat.finditer(text))
        return matches[1].start() if len(matches) > 1 else 0
    
class StructureSplitter:
    def __init__(self, 
                 cleaned_txt_path: pathlib.Path,
                 structure_profile_path: pathlib.Path = pathlib.Path("utils/structure_profile.json"),
                 toc_profile_path: pathlib.Path = pathlib.Path("utils/structure_profile_toc.json"),
                 output_path: pathlib.Path = pathlib.Path("data/paragraphs.jsonl")
                 ):
        """
        Initializes the splitter to divide cleaned book text into structured paragraphs using heading patterns.
        """

        self.cleaned_txt_path = cleaned_txt_path
        self.structure_profile_path = structure_profile_path
        self.toc_profile_path = toc_profile_path
        self.output_path = output_path

    def load_structure_profile(self, profile_path: pathlib.Path) -> Dict:
        """
        Loads or builds a structural profile from the TOC for paragraph splitting.
        """
        if not profile_path.exists():
            print("structure_profile.json missing – building from detected_toc.txt …")
            toc_to_profile = TOCToProfile(toc_profile_path=self.toc_profile_path)
            toc_to_profile.build_profile()
        return json.loads(profile_path.read_text(encoding="utf-8"))

    def split_text_into_units(self, text: str, profile: Dict) -> List[Dict]:
        """
        Splits the book text into paragraph units with structural metadata based on regex heading patterns.
        """
        levels = profile.get("levels", [])
        examples = profile.get("examples", {})

        if not levels or not examples:
            print("[Warning] No valid structure profile found.")
            return []

        results = []
        paragraph_id = 0
        raw_paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

        struct_names = [None for _ in levels]
        struct_ids = [None for _ in levels]
        struct_counters = [0 for _ in levels]

        for para in raw_paragraphs:
            matched_level = None

            for level_idx, level_name in enumerate(levels):
                patterns = examples.get(level_name, [])
                for pattern in patterns:
                    pattern = pattern.strip()
                    try:
                        if re.fullmatch(pattern, para.strip(), flags=re.IGNORECASE):
                            struct_counters[level_idx] += 1
                            struct_names[level_idx] = para.strip()
                            struct_ids[level_idx] = struct_counters[level_idx]

                            for j in range(level_idx + 1, len(levels)):
                                struct_names[j] = None
                                struct_ids[j] = None

                            matched_level = level_idx
                            break
                    except re.error:
                        continue
                if matched_level is not None:
                    break

            if matched_level is None:
                cleaned_text = para.replace("\n", " ").strip()

                low_idx = max((i for i, v in enumerate(struct_names) if v is not None), default=None)
                low_struct = {
                    "index": struct_ids[low_idx] - 1 if low_idx is not None else 0,
                    "name": struct_names[low_idx] if low_idx is not None else "No Section"
                }

                row = {
                    "paragraph_id": paragraph_id,
                    "text": cleaned_text,
                    "low_struct_index": low_struct
                }

                for idx, (val, structid) in enumerate(zip(struct_names, struct_ids)):
                    if val is not None:
                        row[f"struct_{idx}_title"] = val
                        row[f"struct_{idx}_id"] = structid

                results.append(row)
                paragraph_id += 1

        return results

    def run(self, return_paragraphs: bool = False) -> Optional[List[Dict]]:
        """
        Performs the paragraph splitting and saves the output.
        Optionally returns the paragraph list.
        """

        text = self.cleaned_txt_path.read_text(encoding="utf-8")
        detected_toc_path = self.cleaned_txt_path.parent/"detected_toc.txt"

        if detected_toc_path.exists():
            print("Detected TOC file found, using it to build structure profile.")
            profile_path = self.toc_profile_path
        
            if not profile_path.exists():
                TOCToProfile_class = TOCToProfile(
                    detected_toc_path=detected_toc_path,
                    toc_profile_path=self.toc_profile_path
                )
                TOCToProfile_class.build_profile()
        elif self.structure_profile_path.exists():
            print("No TOC found – falling back to default structure profile.")
            profile_path = self.structure_profile_path
        else:
            raise FileNotFoundError(
                f"No detected TOC and no fallback structure profile found.\n"
                f"Missing: {detected_toc_path} and {self.structure_profile_path}"
    )

    
        profile = self.load_structure_profile(profile_path)
        paragraphs = self.split_text_into_units(text, profile)
    
        with self.output_path.open("w", encoding="utf-8") as f:
            for row in paragraphs:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    
        print(f"Saved {len(paragraphs)} structured paragraphs → {self.output_path}")
        return paragraphs if return_paragraphs else None
    

class TOCToProfile:
    def __init__(self,
                 detected_toc_path: pathlib.Path = pathlib.Path("runs/detected_toc.txt"),
                 toc_profile_path: pathlib.Path = pathlib.Path("utils/structure_profile_toc.json")
                 ):
        """
        Initializes the Table of Contents (TOC) profile builder.
        """
        self.detected_toc_path = detected_toc_path
        self.toc_profile_path = toc_profile_path

    def leading_ws(self, line: str) -> int:
        """
        Returns the number of leading whitespace characters in a line.
        """
        return len(line) - len(line.lstrip(" \t"))

    def extract_first_real_heading(self, toc_str: str, exclude: List[str] = None) -> str:
        """
        Finds the first non-excluded heading from the TOC.
        """
        exclude = set(h.lower() for h in (exclude or []))
        lines = [line.strip() for line in toc_str.splitlines()]
        for line in lines:
            if line and line.lower() not in exclude:
                return line
        return ""

    def infer_levels(self, toc_lines: List[str]) -> Dict[str, List[str]]:
        """
        Groups TOC lines into levels based on indentation.
        """
        buckets: Dict[int, List[str]] = {}

        for ln in toc_lines:
            if not ln.strip():
                continue
            indent = self.leading_ws(ln.replace("\t", "    "))
            buckets.setdefault(indent, []).append(ln.strip())

        sorted_depths = sorted(buckets.keys())
        if len(sorted_depths) == 1:
            return {"section": buckets[sorted_depths[0]]}

        levels = {}
        for idx, depth in enumerate(sorted_depths):
            levels[f"level_{idx}"] = buckets[depth]
        return levels

    def build_profile(self):
        """
        Builds a JSON structure profile from detected TOC lines for later paragraph splitting.
        """

        if not self.detected_toc_path.exists():
            print(" === No TOC file found — skipping TOC profile generation.")
            return 

        toc_raw = self.detected_toc_path.read_text(encoding="utf-8").splitlines()
        toc_lines = [ln for ln in toc_raw if ln.strip()]

        clean_lines = []
        dotted = re.compile(r"\.{3,}\s*\d+$")
        for ln in toc_lines:
            ln = dotted.sub("", ln).rstrip()
            if ln:
                clean_lines.append(ln)

        level_dict = self.infer_levels(clean_lines)

        examples = {
            lvl_name: [rf"^{re.escape(h)}$" for h in heads[:50]]
            for lvl_name, heads in level_dict.items()
        }

        profile = {"levels": list(level_dict.keys()), "examples": examples}

        self.toc_profile_path.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")

        print("Structure profile written →", self.toc_profile_path)
        for lvl, heads in level_dict.items():
            print(f"  {lvl}: {len(heads)} headings (showing 3) →", heads[:3])



if __name__ == "__main__":
    pipeline = BookProcessingPipeline(
        raw_txt_path=pathlib.Path("data/republic.txt"),
        book_name="Republic",
        target_speaker="Socrates",
        is_narrator=False
    )
    pipeline.run()