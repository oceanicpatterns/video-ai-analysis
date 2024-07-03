# video-ai-analysis
AI Video Analysis

Step 1.
python3 -m venv .venv

Step 2.
source .venv/bin/activate

Step 3.
pip freeze > requirements.txt

Step 3.
Install from the requirements file

pip install -r requirements.txt


The code captures video frames, processes them to detect pose landmarks using MediaPipe, classifies swimming strokes, analyzes swimming techniques, and generates a PDF report with the analysis results. The report includes detected strokes, technique feedback, and detected equipment.