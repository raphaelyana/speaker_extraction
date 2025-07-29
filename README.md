## Speaker Extraction from Books

This project performs **speaker attribution** in literary texts using a pipeline that combines structure detection, paragraph splitting, and classification via a large language model (Gemini). It is designed to process raw book text and extract who is speaking and narrating or only speaking, with a focus on works like The Republic, Symposium, and Hamlet.

### Project Structure
.
├── pipeline.py                 # Main pipeline script (modular & reusable)
├── execution_file.ipynb        # Interactive notebook to run & explore the pipeline
├── utils/                      # Structure profiles and Table Of Contents extraction logic
├── data/                       # Input texts and generated outputs per book
│   └── [book_name]/           
│       ├── cleaned.txt
│       ├── paragraphs.jsonl
│       └── detected_toc.txt (if applicable)
├── runs/                      # LLM classification outputs and usage reports
│   └── [book_name]/
│       ├── speaker_segments.jsonl
│       └── usage_report.json
└── README.md


---

### How It Works

1. **Filter Book Boundaries**
   * Detects and trims prefaces, intros, and footers
   * Uses LLM to identify the actual narrative start and end
   * Optionally extracts a Table of Contents for structure

2. **Split into Structured Paragraphs**
   * Uses regex-based profiles to identify sections (e.g. Book, Chapter, Scene)
   * Each paragraph is annotated with structural metadata

3. **Classify Speaker/Narrator Paragraphs**
   * Uses Gemini LLM to determine if a paragraph was spoken or narrated by a target character
   * Results saved per book and speaker

---

### Execution

You can run the full pipeline either through:

#### "execution_file.ipynb"

Run and explore results interactively with inline outputs.

#### "pipeline.py"

To run programmatically:

```python
from pipeline import BookProcessingPipeline
import pathlib

pipeline = BookProcessingPipeline(
    raw_txt_path=pathlib.Path("data/republic.txt"),
    book_name="Republic",
    target_speaker="Socrates",
    is_narrator=True
)

pipeline.run()
```

### Requirements
To run the pipeline, you'll need the following Python packages:
```python
python-dotenv
google-generativeai
```
You can install them via pip: 
```python
pip install -r requirements.txt
```

### Environment
Store your API key in a `.env` file as following:
```python
GOOGLE_KEY=your_google_gemini_api_key
```

### Output Example
For a given book like *Republic*, the output includes:
- `data/republic/paragraphs.jsonl`, containing structured text blocks
- `runs/republic/socrates_segments.jsonl`, containing paragraphs attributed to Socrates
- `runs/republic/usage_report.json`, containing LLM token usage tracking

### Notes
- If a book does not contain a Table of Contents, the pipeline falls back to a default structural profile.
- Speaker classification can be configured to detect narration or explicit dialogue only.

### Supported Works (Examples)
- *The Republic*, by Plato
- *Symposium*, by Plato
- *Hamlet*, by Shakespeare
You can easily extend the pipeline to any book in plain text.

### License
This project is licensed under the [Apache License 2.0](./LICENSE).  
You are free to use, modify, and distribute this software for academic or commercial purposes, provided that you give appropriate credit and include the license in any redistribution.  
See the LICENSE file for full terms.
