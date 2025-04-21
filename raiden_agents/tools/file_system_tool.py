import logging
import os
import shutil
from pathlib import Path
import traceback
import datetime
import json
from .base_tool import Tool, ToolExecutionError

logger = logging.getLogger("gemini_agent")

class FileSystemTool(Tool):
    def __init__(self):
        super().__init__(
            name="file_system_operations",
            description="Perform comprehensive operations on the local file system: read/write/append files, create/delete/list directories, move/copy/delete files, get info, check existence.",
            parameters={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": [
                            "read_file", "write_file", "append_to_file",
                            "create_directory", "list_directory", "delete_directory",
                            "delete_file", "move", "copy_file",
                            "get_file_info", "check_exists"
                        ],
                        "description": "The file system operation to perform."
                    },
                    "path": {
                        "type": "string",
                        "description": "The primary file or directory path for the operation."
                    },
                    "destination_path": {
                        "type": "string",
                        "description": "The destination path for 'move' or 'copy_file' operations."
                    },
                    "content": {
                        "type": "string",
                        "description": "The text content for 'write_file' or 'append_to_file'."
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "Flag for recursive operations like 'list_directory' or 'delete_directory' (default: false)."
                    },
                    "force": {
                         "type": "boolean",
                         "description": "Force deletion of non-empty directory (use with caution!) for 'delete_directory' (default: false)."
                    }
                },
                "required": ["operation", "path"]
            }
            # Note: 'content' is conditionally required for write/append
            # Note: 'destination_path' is conditionally required for move/copy
            # Note: 'force'/'recursive' apply only to specific operations
        )

    def _resolve_path(self, path_str, check_exists=False):
        """Resolves path and optionally checks if it exists."""
        if not path_str:
            raise ToolExecutionError("Path parameter cannot be empty.")
        try:
            # Resolve path relative to CWD. Consider security implications.
            resolved = Path(path_str).resolve()
            # Basic check to prevent navigating too far up (adjust as needed)
            # This is a simple check; more robust sandboxing might be needed depending on use case.
            # cwd = Path.cwd().resolve()
            # if cwd not in resolved.parents and resolved != cwd:
            #    logger.warning(f"Attempt to access path outside CWD: {resolved}")
            #    # Decide whether to raise error or allow based on policy
            if check_exists and not resolved.exists():
                 raise ToolExecutionError(f"Path does not exist: {resolved}")
            return resolved
        except Exception as e:
            raise ToolExecutionError(f"Invalid or inaccessible path '{path_str}'. Error: {e}")

    def execute(self, **kwargs):
        # Basic validation handled by superclass, but check conditional requirement
        operation = kwargs.get("operation")
        path_str = kwargs.get("path")
        content = kwargs.get("content") # Optional depending on operation

        if not path_str: # Should be caught by super().validate_args, but double-check
             raise ToolExecutionError("'path' parameter is required.")

        logger.info(f"Executing file system operation '{operation}' on path '{path_str}'")

        try:
            target_path = self._resolve_path(path_str) # Resolve primary path

            # --- Read File ---
            if operation == "read_file":
                target_path = self._resolve_path(path_str, check_exists=True) # Ensure it exists
                if not target_path.is_file():
                    raise ToolExecutionError(f"Path is not a file: {target_path}")
                try:
                    text_content = target_path.read_text(encoding='utf-8')
                    logger.info(f"Successfully read file: {target_path}")
                    return f"Content of '{path_str}':\n```\n{text_content}\n```"
                except Exception as e:
                    raise ToolExecutionError(f"Error reading file '{target_path}': {e}")

            # --- Write File ---
            elif operation == "write_file":
                if content is None: raise ToolExecutionError("'content' is required for 'write_file'.")
                binary_extensions = {'.pdf', '.docx', '.pptx', '.zip', '.jpg', '.jpeg', '.png', '.gif', '.exe', '.dll'}
                warning_note = ""
                if target_path.suffix.lower() in binary_extensions:
                    logger.warning(f"Attempting 'write_file' to binary type '{target_path.suffix}'.")
                    warning_note = "\nNote: Wrote text content to a file with a binary extension. File may not be valid."
                try:
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    target_path.write_text(content, encoding='utf-8')
                    logger.info(f"Successfully wrote content to file: {target_path}")
                    return f"Successfully wrote content to '{path_str}'." + warning_note
                except Exception as e:
                    raise ToolExecutionError(f"Error writing file '{target_path}': {e}")

            # --- Append to File ---
            elif operation == "append_to_file":
                 if content is None: raise ToolExecutionError("'content' is required for 'append_to_file'.")
                 target_path = self._resolve_path(path_str, check_exists=True) # Ensure it exists
                 if not target_path.is_file(): raise ToolExecutionError(f"Path is not a file: {target_path}")
                 try:
                      with target_path.open("a", encoding='utf-8') as f: # Open in append mode
                           f.write(content)
                      logger.info(f"Successfully appended content to file: {target_path}")
                      return f"Successfully appended content to '{path_str}'."
                 except Exception as e:
                      raise ToolExecutionError(f"Error appending to file '{target_path}': {e}")

            # --- Create Directory ---
            elif operation == "create_directory":
                try:
                    target_path.mkdir(parents=True, exist_ok=True)
                    logger.info(f"Ensured directory exists: {target_path}")
                    return f"Directory '{path_str}' ensured/created successfully at {target_path}."
                except Exception as e:
                    raise ToolExecutionError(f"Error creating directory '{target_path}': {e}")

            # --- List Directory ---
            elif operation == "list_directory":
                recursive = kwargs.get("recursive", False)
                target_path = self._resolve_path(path_str, check_exists=True) # Ensure it exists
                if not target_path.is_dir(): raise ToolExecutionError(f"Path is not a directory: {target_path}")
                try:
                    items = []
                    glob_pattern = '**/*' if recursive else '*'
                    for item in sorted(target_path.glob(glob_pattern)):
                         # Avoid listing contents *within* subdirs if not recursive
                         if not recursive and item.parent != target_path:
                              continue
                         item_type = "[DIR] " if item.is_dir() else "[FILE]"
                         # Show path relative to the requested directory for clarity
                         relative_path = item.relative_to(target_path)
                         items.append(f"- {item_type} {relative_path}")

                    if not items and not recursive: # Check direct children if non-recursive
                         if not any(target_path.iterdir()):
                              return f"Directory '{path_str}' is empty."
                    elif not items and recursive: # Check if truly empty if recursive
                         return f"Directory '{path_str}' is empty."

                    return f"Contents of '{path_str}' (Recursive: {recursive}):\n" + "\n".join(items)
                except Exception as e:
                    raise ToolExecutionError(f"Error listing directory '{target_path}': {e}")

            # --- Delete Directory ---
            elif operation == "delete_directory":
                 recursive = kwargs.get("recursive", False)
                 force = kwargs.get("force", False) # Use with extreme caution
                 target_path = self._resolve_path(path_str, check_exists=True) # Ensure it exists
                 if not target_path.is_dir(): raise ToolExecutionError(f"Path is not a directory: {target_path}")

                 try:
                      if not recursive and any(target_path.iterdir()):
                           raise ToolExecutionError(f"Directory '{path_str}' is not empty. Use recursive=true to delete.")
                      if recursive:
                           if not force:
                                # Add a confirmation step or require 'force=true' explicitly
                                raise ToolExecutionError(f"Recursive delete requested for '{path_str}', but 'force=true' was not specified. This is a safety measure. Re-run with force=true if you are sure.")
                           logger.warning(f"Performing FORCE recursive delete on directory: {target_path}")
                           shutil.rmtree(target_path)
                           logger.info(f"Recursively deleted directory: {target_path}")
                           return f"Recursively deleted directory '{path_str}'."
                      else: # Non-recursive, empty directory
                           target_path.rmdir()
                           logger.info(f"Deleted empty directory: {target_path}")
                           return f"Deleted empty directory '{path_str}'."
                 except OSError as e:
                      # Catch specific errors like "Directory not empty" if rmdir fails
                      raise ToolExecutionError(f"Error deleting directory '{target_path}': {e}")
                 except Exception as e:
                      raise ToolExecutionError(f"Error deleting directory '{target_path}': {e}")

            # --- Delete File ---
            elif operation == "delete_file":
                 target_path = self._resolve_path(path_str, check_exists=True) # Ensure it exists
                 if not target_path.is_file(): raise ToolExecutionError(f"Path is not a file: {target_path}")
                 try:
                      target_path.unlink()
                      logger.info(f"Deleted file: {target_path}")
                      return f"Deleted file '{path_str}'."
                 except Exception as e:
                      raise ToolExecutionError(f"Error deleting file '{target_path}': {e}")

            # --- Move/Rename ---
            elif operation == "move":
                 dest_path_str = kwargs.get("destination_path")
                 if not dest_path_str: raise ToolExecutionError("'destination_path' is required for 'move'.")
                 target_path = self._resolve_path(path_str, check_exists=True) # Source must exist
                 destination_path = self._resolve_path(dest_path_str) # Destination might not exist yet
                 try:
                      # Ensure destination parent directory exists
                      destination_path.parent.mkdir(parents=True, exist_ok=True)
                      shutil.move(str(target_path), str(destination_path))
                      logger.info(f"Moved '{target_path}' to '{destination_path}'")
                      return f"Moved '{path_str}' to '{dest_path_str}'."
                 except Exception as e:
                      raise ToolExecutionError(f"Error moving '{target_path}' to '{destination_path}': {e}")

            # --- Copy File ---
            elif operation == "copy_file":
                 dest_path_str = kwargs.get("destination_path")
                 if not dest_path_str: raise ToolExecutionError("'destination_path' is required for 'copy_file'.")
                 target_path = self._resolve_path(path_str, check_exists=True) # Source must exist
                 if not target_path.is_file(): raise ToolExecutionError(f"Source path is not a file: {target_path}")
                 destination_path = self._resolve_path(dest_path_str) # Destination might not exist yet
                 try:
                      # Ensure destination parent directory exists
                      destination_path.parent.mkdir(parents=True, exist_ok=True)
                      shutil.copy2(str(target_path), str(destination_path)) # copy2 preserves metadata
                      logger.info(f"Copied '{target_path}' to '{destination_path}'")
                      return f"Copied file '{path_str}' to '{dest_path_str}'."
                 except Exception as e:
                      raise ToolExecutionError(f"Error copying '{target_path}' to '{destination_path}': {e}")

            # --- Get File Info ---
            elif operation == "get_file_info":
                 target_path = self._resolve_path(path_str, check_exists=True) # Ensure it exists
                 try:
                      stat_info = target_path.stat()
                      info = {
                           "path": str(target_path),
                           "type": "directory" if target_path.is_dir() else "file",
                           "size_bytes": stat_info.st_size,
                           "last_modified": datetime.datetime.fromtimestamp(stat_info.st_mtime).isoformat(),
                           "created": datetime.datetime.fromtimestamp(stat_info.st_ctime).isoformat(),
                           # Permissions might be OS-specific, skipping for simplicity
                      }
                      return f"File info for '{path_str}':\n" + json.dumps(info, indent=2)
                 except Exception as e:
                      raise ToolExecutionError(f"Error getting info for '{target_path}': {e}")

            # --- Check Exists ---
            elif operation == "check_exists":
                 # _resolve_path doesn't check existence by default here
                 exists = target_path.exists()
                 item_type = "Unknown"
                 if exists:
                      item_type = "directory" if target_path.is_dir() else "file"
                 return f"Path '{path_str}' exists: {exists} (Type: {item_type})"

            else:
                raise ToolExecutionError(f"Unsupported file system operation: {operation}")

        except ToolExecutionError as e:
            raise e # Re-raise specific tool errors
        except Exception as e:
            logger.error(f"Unexpected error in FileSystemTool operation '{operation}' on path '{path_str}': {e}", exc_info=True)
            traceback.print_exc()
            raise ToolExecutionError(f"An unexpected error occurred: {e}")
