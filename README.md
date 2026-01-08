# OCR Claim Forms – HCFA-1500 & UB-04

This project focuses on extracting key information from **HCFA-1500 (CMS-1500)** and **UB-04** healthcare claim forms using **Optical Character Recognition (OCR)** and **Natural Language Processing (NLP)**.

The input to the system is a clear, typed image (PNG) of a claim form. OCR is used to convert the image into text, and an NLP pipeline is then applied to identify and extract relevant information from the extracted text.

From **HCFA-1500 (CMS-1500-1)** forms, the following details are extracted:
- Patient name  
- Date of birth  
- ICD-10 diagnosis codes  
- CPT/HCPCS procedure codes  
- Charge amounts  

From **UB-04 (CMS-1450)** forms, the following details are extracted:
- Patient name  
- Admission date  
- Principal diagnosis  
- Billing provider NPI  
- Total charges  

This project serves as a simple proof of concept demonstrating how OCR and NLP techniques can be applied to structured healthcare documents to convert unstructured claim form images into structured data. It can be extended further to support additional fields, improved accuracy, or other claim form types.

For best results, the claim form images should be high quality, properly aligned, and contain typed text.
