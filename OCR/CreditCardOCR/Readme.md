# Objective: Credit Card OCR with OpenCV and Python
Applying template matching as a form of OCR to help us create a solution to automatically recognize credit cards and extract the associated credit card digits from images.

Activities:
* Detect the location of the credit card in the image.
* Localize the four groupings of four digits, pertaining to the sixteen digits on the credit card.
* Apply OCR to recognize the sixteen digits on the credit card.
* Recognize the type of credit card (i.e., Visa, MasterCard, American Express, etc.).

We devise a computer vision and image processing algorithm that can:
* Create dictionary of OCR-A Font letters.
* Localize the four groupings of four digits on a credit card.
* Extract each of these four groupings followed by segmenting each of the sixteen numbers individually.
* Recognize each of the sixteen credit card digits by using template matching and the OCR-A font.

## Usage
> python CreditcardOCR.py --image ./cc_1.png --reference ./ocr_a_reference.png