import logging
import os
from pathlib import Path
import traceback
from PyPDF2 import PdfReader, PdfWriter, PdfMerger
from .base_tool import Tool, ToolExecutionError
import re # For parsing page ranges

logger = logging.getLogger("gemini_agent")

class PdfTool(Tool):
    def __init__(self):
        super().__init__(
            name="pdf_operations",
            description="Perform comprehensive operations on PDF files: extract text/metadata, get page count, merge, split, rotate, add watermark, encrypt/decrypt.",
            parameters={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": [
                            "extract_text", "get_metadata", "get_page_count",
                            "merge_pdfs", "split_pdf", "rotate_pages",
                            "add_watermark", "encrypt_pdf", "decrypt_pdf"
                         ],
                        "description": "The PDF operation to perform."
                    },
                    "input_path": {
                        "type": "string",
                        "description": "The path to the primary input PDF file."
                    },
                    "input_paths": { # For merge_pdfs
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of input PDF paths for merging."
                    },
                    "output_path": { # For merge, rotate, watermark, encrypt, decrypt
                        "type": "string",
                        "description": "Path for the output PDF file."
                    },
                    "output_directory": { # For split_pdf
                        "type": "string",
                        "description": "Directory to save split pages."
                    },
                    "watermark_path": { # For add_watermark
                        "type": "string",
                        "description": "Path to the PDF file to use as a watermark (first page is used)."
                    },
                    "page_numbers": { # For extract_text, rotate_pages
                        "type": "string",
                        "description": "Page numbers/ranges (e.g., '1,3,5-7'). Default all for extract."
                    },
                    "angle": { # For rotate_pages
                        "type": "integer",
                        "description": "Rotation angle (multiple of 90)."
                    },
                    "password": { # For encrypt_pdf, decrypt_pdf
                         "type": "string",
                         "description": "Password for encryption or decryption."
                    },
                    # PyPDF2 encrypt supports separate user/owner passwords, but let's keep it simple
                    # "user_password": {
                    #      "type": "string",
                    #      "description": "User password for encryption."
                    # },
                    # "owner_password": {
                    #      "type": "string",
                    #      "description": "Owner password for encryption (optional)."
                    # }
                },
                "required": ["operation"] # Other params conditionally required
            }
        )

    def _resolve_path(self, path_str, check_exists=False, check_is_pdf=False):
        """Resolves path, optionally checks existence and PDF extension."""
        if not path_str:
            raise ToolExecutionError("Path parameter cannot be empty.")
        try:
            resolved = Path(path_str).resolve()
            if check_exists and not resolved.exists():
                 raise ToolExecutionError(f"Path does not exist: {resolved}")
            if check_is_pdf and resolved.suffix.lower() != '.pdf':
                 raise ToolExecutionError(f"Path is not a PDF file: {resolved}")
            return resolved
        except ToolExecutionError as e:
             raise e # Re-raise specific errors
        except Exception as e:
            raise ToolExecutionError(f"Invalid or inaccessible path '{path_str}'. Error: {e}")

    def _parse_page_numbers(self, page_str, max_pages):
        """Parses a string like '1,3,5-7' into a set of 0-based page indices."""
        if not page_str:
            return set(range(max_pages)) # Default to all pages if none specified

        pages = set()
        try:
            parts = page_str.split(',')
            for part in parts:
                part = part.strip()
                if '-' in part:
                    start, end = map(int, part.split('-'))
                    if start < 1 or end > max_pages or start > end:
                        raise ValueError(f"Invalid page range: {part}")
                    pages.update(range(start - 1, end)) # 0-based index
                else:
                    page_num = int(part)
                    if page_num < 1 or page_num > max_pages:
                        raise ValueError(f"Invalid page number: {page_num}")
                    pages.add(page_num - 1) # 0-based index
            return pages
        except ValueError as e:
            raise ToolExecutionError(f"Invalid page_numbers format '{page_str}'. Use comma-separated numbers/ranges (e.g., '1,3,5-7'). Error: {e}")

    def execute(self, **kwargs):
        operation = kwargs.get("operation")
        input_path_str = kwargs.get("input_path")
        input_paths_list = kwargs.get("input_paths")
        output_path_str = kwargs.get("output_path")
        output_dir_str = kwargs.get("output_directory")
        page_numbers_str = kwargs.get("page_numbers")
        angle_str = kwargs.get("angle")
        password = kwargs.get("password")
        watermark_path_str = kwargs.get("watermark_path")

        logger.info(f"Executing PDF operation: {operation}")

        try:
            # --- Operation: extract_text ---
            if operation == "extract_text":
                if not input_path_str: raise ToolExecutionError("'input_path' is required.")
                input_path = self._resolve_path(input_path_str, check_exists=True, check_is_pdf=True)

                reader = PdfReader(input_path)
                if reader.is_encrypted:
                     logger.warning(f"PDF '{input_path}' is encrypted. Attempting decryption with empty password.")
                     # Try decrypting with an empty password first, common for some PDFs
                     if reader.decrypt("") == 0: # 0 means failure
                          raise ToolExecutionError(f"Cannot extract text from encrypted PDF '{input_path_str}'. Decryption failed (try providing password if known via decrypt_pdf operation first).")
                     # If decrypt succeeds with empty password, proceed

                max_pages = len(reader.pages)
                target_pages = self._parse_page_numbers(page_numbers_str, max_pages)

                extracted_text = ""
                for i in sorted(list(target_pages)): # Process in page order
                    try:
                        page = reader.pages[i]
                        extracted_text += f"--- Page {i+1} ---\n"
                        extracted_text += page.extract_text() + "\n"
                    except Exception as page_e:
                         logger.warning(f"Could not extract text from page {i+1} of {input_path}: {page_e}")
                         extracted_text += f"--- Page {i+1} (Error extracting text) ---\n"

                logger.info(f"Extracted text from {len(target_pages)} pages of {input_path}")
                return f"Extracted text from '{input_path_str}' (Pages: {page_numbers_str or 'all'}):\n```\n{extracted_text}\n```"

            # --- Operation: get_metadata ---
            elif operation == "get_metadata":
                if not input_path_str: raise ToolExecutionError("'input_path' is required.")
                input_path = self._resolve_path(input_path_str, check_exists=True, check_is_pdf=True)

                reader = PdfReader(input_path)
                if reader.is_encrypted:
                     # Try decrypting with an empty password first
                     if reader.decrypt("") == 0:
                          raise ToolExecutionError(f"Cannot get metadata from encrypted PDF '{input_path_str}'. Decryption failed.")
                metadata = reader.metadata
                if not metadata:
                    return f"No metadata found in '{input_path_str}'."

                # Format metadata nicely
                meta_dict = {k: v for k, v in metadata.items()}
                return f"Metadata for '{input_path_str}':\n" + json.dumps(meta_dict, indent=2)

            # --- Operation: get_page_count ---
            elif operation == "get_page_count":
                if not input_path_str: raise ToolExecutionError("'input_path' is required.")
                input_path = self._resolve_path(input_path_str, check_exists=True, check_is_pdf=True)

                reader = PdfReader(input_path)
                if reader.is_encrypted:
                     if reader.decrypt("") == 0:
                          raise ToolExecutionError(f"Cannot get page count from encrypted PDF '{input_path_str}'. Decryption failed.")
                count = len(reader.pages)
                return f"The PDF '{input_path_str}' has {count} pages."

            # --- Operation: merge_pdfs ---
            elif operation == "merge_pdfs":
                if not input_paths_list or not isinstance(input_paths_list, list):
                    raise ToolExecutionError("'input_paths' (a list of PDF paths) is required.")
                if not output_path_str: raise ToolExecutionError("'output_path' is required.")

                output_path = self._resolve_path(output_path_str)
                if output_path.suffix.lower() != '.pdf':
                     output_path = output_path.with_suffix('.pdf')

                merger = PdfMerger()
                resolved_input_paths = []
                for p_str in input_paths_list:
                    p = self._resolve_path(p_str, check_exists=True, check_is_pdf=True)
                    # Check for encryption before merging
                    reader_check = PdfReader(p)
                    if reader_check.is_encrypted:
                         if reader_check.decrypt("") == 0: # Try empty password
                              merger.close() # Close merger if error occurs
                              raise ToolExecutionError(f"Cannot merge encrypted PDF '{p_str}'. Decrypt it first.")
                    merger.append(str(p))
                    resolved_input_paths.append(p)

                output_path.parent.mkdir(parents=True, exist_ok=True)
                merger.write(str(output_path))
                merger.close()
                logger.info(f"Merged {len(resolved_input_paths)} PDFs into {output_path}")
                return f"Successfully merged {len(resolved_input_paths)} PDFs into '{output_path_str}'."

            # --- Operation: split_pdf ---
            elif operation == "split_pdf":
                if not input_path_str: raise ToolExecutionError("'input_path' is required.")
                if not output_dir_str: raise ToolExecutionError("'output_directory' is required.")

                input_path = self._resolve_path(input_path_str, check_exists=True, check_is_pdf=True)
                output_dir = self._resolve_path(output_dir_str)

                output_dir.mkdir(parents=True, exist_ok=True)

                reader = PdfReader(input_path)
                if reader.is_encrypted:
                     if reader.decrypt("") == 0:
                          raise ToolExecutionError(f"Cannot split encrypted PDF '{input_path_str}'. Decrypt it first.")
                num_pages = len(reader.pages)
                base_filename = input_path.stem

                for i in range(num_pages):
                    writer = PdfWriter()
                    writer.add_page(reader.pages[i])
                    output_filename = output_dir / f"{base_filename}_page_{i+1}.pdf"
                    with open(output_filename, "wb") as output_pdf:
                        writer.write(output_pdf)

                logger.info(f"Split '{input_path}' into {num_pages} pages in directory '{output_dir}'")
                return f"Successfully split '{input_path_str}' into {num_pages} pages in the directory '{output_dir_str}'."

            # --- Operation: rotate_pages ---
            elif operation == "rotate_pages":
                if not input_path_str: raise ToolExecutionError("'input_path' is required.")
                if not output_path_str: raise ToolExecutionError("'output_path' is required.")
                if not page_numbers_str: raise ToolExecutionError("'page_numbers' string is required.")
                if angle_str is None: raise ToolExecutionError("'angle' (multiple of 90) is required.")

                input_path = self._resolve_path(input_path_str, check_exists=True, check_is_pdf=True)
                output_path = self._resolve_path(output_path_str)
                if output_path.suffix.lower() != '.pdf':
                     output_path = output_path.with_suffix('.pdf')

                try:
                    angle = int(angle_str)
                    if angle % 90 != 0: raise ValueError("Angle must be a multiple of 90.")
                except ValueError as e:
                    raise ToolExecutionError(f"Invalid angle '{angle_str}'. Error: {e}")

                reader = PdfReader(input_path)
                if reader.is_encrypted:
                     if reader.decrypt("") == 0:
                          raise ToolExecutionError(f"Cannot rotate pages in encrypted PDF '{input_path_str}'. Decrypt it first.")

                writer = PdfWriter()
                max_pages = len(reader.pages)
                target_pages = self._parse_page_numbers(page_numbers_str, max_pages)

                for i in range(max_pages):
                    page = reader.pages[i]
                    if i in target_pages:
                        page.rotate(angle)
                    writer.add_page(page)

                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "wb") as output_pdf:
                    writer.write(output_pdf)

                logger.info(f"Rotated pages {page_numbers_str} by {angle} degrees in '{input_path}' and saved to '{output_path}'")
                return f"Successfully rotated pages '{page_numbers_str}' by {angle} degrees in '{input_path_str}' and saved to '{output_path_str}'."

            # --- Operation: add_watermark ---
            elif operation == "add_watermark":
                 if not input_path_str: raise ToolExecutionError("'input_path' is required.")
                 if not watermark_path_str: raise ToolExecutionError("'watermark_path' (PDF) is required.")
                 if not output_path_str: raise ToolExecutionError("'output_path' is required.")

                 input_path = self._resolve_path(input_path_str, check_exists=True, check_is_pdf=True)
                 watermark_path = self._resolve_path(watermark_path_str, check_exists=True, check_is_pdf=True)
                 output_path = self._resolve_path(output_path_str)
                 if output_path.suffix.lower() != '.pdf': output_path = output_path.with_suffix('.pdf')

                 reader = PdfReader(input_path)
                 if reader.is_encrypted:
                      if reader.decrypt("") == 0: raise ToolExecutionError(f"Cannot watermark encrypted PDF '{input_path_str}'. Decrypt it first.")

                 watermark_reader = PdfReader(watermark_path)
                 watermark_page = watermark_reader.pages[0] # Use first page of watermark PDF

                 writer = PdfWriter()

                 for page in reader.pages:
                      page.merge_page(watermark_page) # Merge watermark onto each page
                      writer.add_page(page)

                 output_path.parent.mkdir(parents=True, exist_ok=True)
                 with open(output_path, "wb") as output_pdf:
                      writer.write(output_pdf)

                 logger.info(f"Added watermark from '{watermark_path}' to '{input_path}' and saved to '{output_path}'")
                 return f"Successfully added watermark from '{watermark_path_str}' to '{input_path_str}' and saved as '{output_path_str}'."

            # --- Operation: encrypt_pdf ---
            elif operation == "encrypt_pdf":
                 if not input_path_str: raise ToolExecutionError("'input_path' is required.")
                 if not output_path_str: raise ToolExecutionError("'output_path' is required.")
                 if not password: raise ToolExecutionError("'password' is required for encryption.")

                 input_path = self._resolve_path(input_path_str, check_exists=True, check_is_pdf=True)
                 output_path = self._resolve_path(output_path_str)
                 if output_path.suffix.lower() != '.pdf': output_path = output_path.with_suffix('.pdf')

                 reader = PdfReader(input_path)
                 if reader.is_encrypted: # Avoid re-encrypting
                      raise ToolExecutionError(f"Input PDF '{input_path_str}' is already encrypted.")

                 writer = PdfWriter()
                 for page in reader.pages: # Copy all pages
                      writer.add_page(page)

                 # Encrypt with the provided password (acts as both user and owner password here)
                 writer.encrypt(password)

                 output_path.parent.mkdir(parents=True, exist_ok=True)
                 with open(output_path, "wb") as output_pdf:
                      writer.write(output_pdf)

                 logger.info(f"Encrypted '{input_path}' and saved to '{output_path}'")
                 return f"Successfully encrypted '{input_path_str}' and saved as '{output_path_str}'."

            # --- Operation: decrypt_pdf ---
            elif operation == "decrypt_pdf":
                 if not input_path_str: raise ToolExecutionError("'input_path' is required.")
                 if not output_path_str: raise ToolExecutionError("'output_path' is required.")
                 if not password: raise ToolExecutionError("'password' is required for decryption.")

                 input_path = self._resolve_path(input_path_str, check_exists=True, check_is_pdf=True)
                 output_path = self._resolve_path(output_path_str)
                 if output_path.suffix.lower() != '.pdf': output_path = output_path.with_suffix('.pdf')

                 reader = PdfReader(input_path)
                 if not reader.is_encrypted:
                      return f"File '{input_path_str}' is not encrypted. No decryption needed. (Copying to output path)."
                      # Optionally just copy the file if not encrypted
                      # shutil.copy2(input_path, output_path)
                      # return f"File '{input_path_str}' was not encrypted. Copied to '{output_path_str}'."

                 # Attempt decryption
                 if reader.decrypt(password) == 0: # 0 indicates failure
                      raise ToolExecutionError(f"Decryption failed for '{input_path_str}'. Incorrect password?")
                 # Decryption successful if decrypt returns 1 or 2

                 writer = PdfWriter()
                 for page in reader.pages: # Copy decrypted pages
                      writer.add_page(page)

                 output_path.parent.mkdir(parents=True, exist_ok=True)
                 with open(output_path, "wb") as output_pdf:
                      writer.write(output_pdf)

                 logger.info(f"Decrypted '{input_path}' and saved to '{output_path}'")
                 return f"Successfully decrypted '{input_path_str}' and saved as '{output_path_str}'."

            else:
                raise ToolExecutionError(f"Unsupported PDF operation: {operation}")

        except ImportError:
             logger.error("PyPDF2 is required for PDF operations but not installed.")
             raise ToolExecutionError("PDF processing library (PyPDF2) not installed.")
        except FileNotFoundError as e:
             # Catch specific file not found errors from Path operations
             raise ToolExecutionError(f"File not found: {e}")
        except ToolExecutionError as e:
             raise e # Re-raise specific tool errors
        except Exception as e:
            logger.error(f"Unexpected error during PDF operation '{operation}': {e}", exc_info=True)
            traceback.print_exc()
            # Check for common PyPDF2 errors
            err_str = str(e).lower()
            if "encrypted" in err_str and "password" not in err_str: # Check if it's an encryption error *before* password attempt
                 raise ToolExecutionError(f"PDF operation failed: File '{input_path_str}' might be encrypted and requires decryption first.")
            raise ToolExecutionError(f"An unexpected error occurred during the PDF operation: {e}")
