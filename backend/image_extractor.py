import os
import io

def get_pdf_page_image(pdf_path: str, page_number: int) -> bytes:
    """
    Renders a specific page of a PDF file to a PNG image using PyMuPDF (fitz).
    Returns the binary image data (bytes), or None if it fails.
    """
    try:
        import fitz
    except ImportError:
        print("PyMuPDF (fitz) is not installed. Will not render PDF images.")
        return None
        
    if not os.path.exists(pdf_path):
        print(f"Error: PDF not found at {pdf_path}")
        return None
        
    try:
        # Load PDF document
        doc = fitz.open(pdf_path)
        
        # Validate page number bounds
        if page_number < 0 or page_number >= len(doc):
            return None
            
        # Load exactly the required page
        page = doc.load_page(page_number)
        
        # Increase DPI to 150 (matrix zoom = 2.0 or so) to make textbooks very readable
        mat = fitz.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=mat)
        
        # Export as PNG bytes natively
        img_bytes = pix.tobytes("png")
        doc.close()
        
        return img_bytes
    except Exception as e:
        print(f"Failed to extract image from PDF: {str(e)}")
        return None
