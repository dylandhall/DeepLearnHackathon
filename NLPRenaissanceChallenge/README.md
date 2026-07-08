# RenAIssance: OCR for Seventeenth Century Spanish Texts
<!-- ![coverImage](./cover.webp) -->
<p align="center">
<img src="images/cover.webp" width="400">
</p>

## Description
RenAIssance project encompasses the use of optical character recognition (OCR) necessary to digitize text sources that have not yet been targeted by existing tools. Its purpose is to find ways to apply machine learning techniques to make OCR possible on a variety of materials that have never been digitized before. This competition focuses on applying AI to recognize text in Spanish printed sources from the seventeenth century, an area underserved by current OCR tools. 
**For further reference, you can go through this presentation:** [Description of the file](data/Hackathon%20training%20-%20print%20irregularities.pptx)
 

Transliteration of text from centuries-old works represents a research area that is underserved by current tools, such as Adobe Acrobat’s OCR. While these resources can perform text recognition from clearly printed modern sources, they are incapable of extracting textual data from early forms of print, much less manuscripts.

## Task
Build a model for optically recognizing the text in the given image dataset. Pick the most appropriate approach and discuss your strategy.

## Dataset
- The dataset consists of two scanned Spanish textbooks with 31 pages each.
- Transcriptions of the first 25 pages (ground truth) are available in a DOCX file.
- The first 25 pages will serve as your training and evaluation data.
- You need to submit the predicted text for the remaining 6 pages as the output of your model for testing purposes.

## Evaluation Metrics
- Please see the attached PDF with the exact evaluation metrics and the evaluation pipeline.

## Deliverables
- Fill out the pre- and post-hackathon surveys
- A trained OCR model capable of recognizing text in the provided datasets.
- Predictions of the last 6 pages of the provided books (SEE EVALUATION GUIDELINES)
- A brief report discussing the chosen approach, model architecture, training process, and evaluation results.
- Example usage notebook demonstrating the model's performance on sample data.

## Example Notebook
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1oHxG16Taz98C74znJjEcqRlvYjU7KKAe?usp=sharing)

## Contributors

- Shashank Shekhar Singh (Indian Institue of Technology, BHU)
- Utsav Rai (Imperial College London)
- Kate O'Reilly (Trinity College Dublin)
- Yukinori Yamamoto (Waseda University)
- Pranav Kulkarni (Indian Institute of Technology, Bombay)
- Arsh Khan (VJTI, Mumbai)
- Saarthak Gupta (Indian Institute of Technology, BHU)
- Xabier Granja (University of Alabama)
- Sergei Gleyzer (University of Alabama)
- Harrison Meadows (University of Tennessee)
