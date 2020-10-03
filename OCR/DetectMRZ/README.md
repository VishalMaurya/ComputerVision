# Objective: Detect Machine-readable Zones (MRZs) 
Detect Machine-readable Zones (MRZs) in passport scans using only basic image processing techniques, namely:
    * Thresholding
    * Gradients
    * Morphological operations (specifically, closings and erosions)
    * Contour properties

These operations, while simple, allowed us to detect the MRZ regions in images without having to rely on more advanced feature extraction and machine learning methods such as Linear SVM + HOG for object detection.