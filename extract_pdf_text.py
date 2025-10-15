import PyPDF2
import os

def extract_text_from_pdf(pdf_path, txt_path):
    """
    Extracts text from a PDF file and saves it to a text file.
    """
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at '{pdf_path}'")
        return False
        
    try:
        with open(pdf_path, 'rb') as pdf_file, open(txt_path, 'w', encoding='utf-8') as txt_file:
            reader = PyPDF2.PdfReader(pdf_file)
            print(f"Reading PDF with {len(reader.pages)} pages...")
            
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text = page.extract_text()
                if text:
                    txt_file.write(f"--- Page {page_num + 1} ---\n")
                    txt_file.write(text)
                    txt_file.write("\n\n")
            
        print(f"Text successfully extracted to '{txt_path}'")
        return True
    except Exception as e:
        print(f"An error occurred during PDF text extraction: {e}")
        return False

if __name__ == '__main__':
    pdf_filename = "2010-A high accuracy conformal method for evaluating the discontinuous Fourier transform(1).pdf"
    txt_filename = "paper_content.txt"
    extract_text_from_pdf(pdf_filename, txt_filename) 