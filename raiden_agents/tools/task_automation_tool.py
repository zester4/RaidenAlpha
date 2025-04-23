import logging
import os
import subprocess
import json
import time
from pathlib import Path
from datetime import datetime
from .base_tool import Tool, ToolExecutionError

logger = logging.getLogger("raiden_agent")

class TaskAutomationTool(Tool):
    def __init__(self):
        super().__init__(
            name="automate_task",
            description="Automate repetitive tasks using native system commands and scripting",
            parameters={
                "type": "object",
                "properties": {
                    "task_type": {
                        "type": "string",
                        "description": "Type of task to automate (file_management, system_cleanup, batch_processing, backup)",
                        "enum": ["file_management", "system_cleanup", "batch_processing", "backup"]
                    },
                    "task_parameters": {
                        "type": "object",
                        "description": "Parameters specific to the task type"
                    },
                    "schedule": {
                        "type": "string", 
                        "description": "Optional: When to run this task (now, hourly, daily, weekly)",
                        "enum": ["now", "hourly", "daily", "weekly"]
                    }
                },
                "required": ["task_type", "task_parameters"]
            }
        )
        # Create task history directory
        self.task_history_dir = Path("task_automation_history")
        os.makedirs(self.task_history_dir, exist_ok=True)
    
    def execute(self, **kwargs):
        self.validate_args(kwargs)
        
        try:
            task_type = kwargs.get("task_type")
            task_parameters = kwargs.get("task_parameters")
            schedule = kwargs.get("schedule", "now")
            
            logger.info(f"Automating task of type: {task_type} with schedule: {schedule}")
            
            # Generate a unique task ID
            task_id = f"{task_type}_{int(time.time())}"
            
            # Record task initiation
            self._record_task_event(task_id, "started", task_parameters)
            
            # If not running now, schedule for later
            if schedule != "now":
                return self._schedule_task(task_id, task_type, task_parameters, schedule)
            
            # Execute the appropriate task based on type
            if task_type == "file_management":
                result = self._handle_file_management(task_parameters)
            elif task_type == "system_cleanup":
                result = self._handle_system_cleanup(task_parameters)
            elif task_type == "batch_processing":
                result = self._handle_batch_processing(task_parameters)
            elif task_type == "backup":
                result = self._handle_backup(task_parameters)
            else:
                raise ToolExecutionError(f"Unsupported task type: {task_type}")
            
            # Record task completion
            self._record_task_event(task_id, "completed", result)
            
            # Store in vector DB if available
            try:
                from __main__ import vector_db
                if vector_db.is_ready():
                    vector_db.add(
                        f"Automated task: {task_type}",
                        {
                            "type": "automated_task",
                            "task_id": task_id,
                            "task_type": task_type,
                            "parameters": task_parameters,
                            "result": result,
                            "time": datetime.now().isoformat()
                        }
                    )
            except ImportError:
                pass
            
            return f"Task {task_id} completed successfully: {result}"
            
        except Exception as e:
            logger.error(f"Task automation error: {e}", exc_info=True)
            raise ToolExecutionError(f"Failed to automate task: {e}")
    
    def _handle_file_management(self, parameters):
        """Handle file management tasks like organizing, renaming, moving files"""
        operation = parameters.get("operation", "organize")
        source_dir = parameters.get("source_directory", ".")
        target_dir = parameters.get("target_directory")
        file_pattern = parameters.get("file_pattern", "*")
        
        source_path = Path(source_dir)
        if not source_path.exists():
            raise ToolExecutionError(f"Source directory does not exist: {source_dir}")
        
        if operation == "organize":
            # Create organization structure
            if target_dir:
                os.makedirs(target_dir, exist_ok=True)
            else:
                target_dir = source_dir
            
            # Find and organize files
            file_count = 0
            for file_path in source_path.glob(file_pattern):
                if file_path.is_file():
                    # Determine category based on extension
                    extension = file_path.suffix.lower()[1:]
                    if extension in ["jpg", "jpeg", "png", "gif"]:
                        category = "images"
                    elif extension in ["mp3", "wav", "flac"]:
                        category = "audio"
                    elif extension in ["mp4", "avi", "mov"]:
                        category = "videos"
                    elif extension in ["pdf", "doc", "docx", "txt"]:
                        category = "documents"
                    else:
                        category = "other"
                    
                    # Create category directory
                    category_dir = Path(target_dir) / category
                    os.makedirs(category_dir, exist_ok=True)
                    
                    # Move file
                    target_path = category_dir / file_path.name
                    if not target_path.exists():
                        file_path.rename(target_path)
                        file_count += 1
            
            return f"Organized {file_count} files into categories in {target_dir}"
        
        elif operation == "rename":
            prefix = parameters.get("prefix", "")
            suffix = parameters.get("suffix", "")
            
            file_count = 0
            for file_path in source_path.glob(file_pattern):
                if file_path.is_file():
                    new_name = f"{prefix}{file_path.stem}{suffix}{file_path.suffix}"
                    file_path.rename(file_path.parent / new_name)
                    file_count += 1
            
            return f"Renamed {file_count} files in {source_dir}"
        
        else:
            raise ToolExecutionError(f"Unsupported file operation: {operation}")
    
    def _handle_system_cleanup(self, parameters):
        """Handle system cleanup tasks like removing temporary files"""
        cleanup_type = parameters.get("cleanup_type", "temp_files")
        target_dir = parameters.get("target_directory")
        
        if cleanup_type == "temp_files":
            # Default temp locations based on OS
            if not target_dir:
                if os.name == "nt":  # Windows
                    target_dir = os.environ.get("TEMP")
                else:  # Linux/Mac
                    target_dir = "/tmp"
            
            target_path = Path(target_dir)
            if not target_path.exists():
                return f"Target directory does not exist: {target_dir}"
            
            # Find and remove files older than specified days
            days_old = parameters.get("days_old", 7)
            current_time = time.time()
            max_age = days_old * 86400  # Convert days to seconds
            
            removed_count = 0
            skipped_count = 0
            
            for file_path in target_path.glob("*"):
                try:
                    if file_path.is_file():
                        file_age = current_time - file_path.stat().st_mtime
                        if file_age > max_age:
                            file_path.unlink()
                            removed_count += 1
                        else:
                            skipped_count += 1
                except (PermissionError, OSError):
                    skipped_count += 1
            
            return f"Cleaned up {removed_count} temporary files (skipped {skipped_count})"
        
        elif cleanup_type == "empty_directories":
            if not target_dir:
                raise ToolExecutionError("Target directory must be specified for empty_directories cleanup")
            
            target_path = Path(target_dir)
            if not target_path.exists():
                return f"Target directory does not exist: {target_dir}"
            
            removed_count = 0
            
            # First pass to identify empty directories
            empty_dirs = []
            for root, dirs, files in os.walk(target_path, topdown=False):
                for dir_name in dirs:
                    dir_path = Path(root) / dir_name
                    if not any(dir_path.iterdir()):
                        empty_dirs.append(dir_path)
            
            # Remove empty directories
            for dir_path in empty_dirs:
                try:
                    dir_path.rmdir()
                    removed_count += 1
                except (PermissionError, OSError):
                    pass
            
            return f"Removed {removed_count} empty directories"
        
        else:
            raise ToolExecutionError(f"Unsupported cleanup type: {cleanup_type}")
    
    def _handle_batch_processing(self, parameters):
        """Handle batch processing of files or commands"""
        processing_type = parameters.get("processing_type", "commands")
        
        if processing_type == "commands":
            commands = parameters.get("commands", [])
            if not commands:
                raise ToolExecutionError("No commands specified for batch processing")
            
            results = []
            for cmd in commands:
                try:
                    logger.info(f"Executing command: {cmd}")
                    # Execute command and capture output
                    result = subprocess.run(
                        cmd, 
                        shell=True, 
                        capture_output=True, 
                        text=True,
                        timeout=parameters.get("timeout", 60)
                    )
                    
                    status = "success" if result.returncode == 0 else "failed"
                    results.append({
                        "command": cmd,
                        "status": status,
                        "return_code": result.returncode,
                        "output": result.stdout[:1000],  # Limit output size
                        "error": result.stderr[:1000] if result.stderr else None
                    })
                    
                except subprocess.TimeoutExpired:
                    results.append({
                        "command": cmd,
                        "status": "timeout",
                        "error": "Command execution timed out"
                    })
                except Exception as e:
                    results.append({
                        "command": cmd,
                        "status": "error",
                        "error": str(e)
                    })
            
            successful = sum(1 for r in results if r["status"] == "success")
            return f"Executed {len(commands)} commands ({successful} successful)"
        
        elif processing_type == "file_conversion":
            source_dir = parameters.get("source_directory", ".")
            file_pattern = parameters.get("file_pattern", "*")
            conversion_type = parameters.get("conversion_type")
            
            if not conversion_type:
                raise ToolExecutionError("No conversion type specified")
            
            source_path = Path(source_dir)
            if not source_path.exists():
                raise ToolExecutionError(f"Source directory does not exist: {source_dir}")
            
            processed = 0
            
            for file_path in source_path.glob(file_pattern):
                if file_path.is_file():
                    try:
                        # Handle different conversion types
                        if conversion_type == "text_to_uppercase":
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                            
                            with open(file_path, 'w', encoding='utf-8') as f:
                                f.write(content.upper())
                            processed += 1
                            
                        elif conversion_type == "text_to_lowercase":
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                            
                            with open(file_path, 'w', encoding='utf-8') as f:
                                f.write(content.lower())
                            processed += 1
                            
                        elif conversion_type == "line_count":
                            with open(file_path, 'r', encoding='utf-8') as f:
                                line_count = sum(1 for _ in f)
                            
                            # Append line count to filename
                            stats_dir = source_path / "file_stats"
                            os.makedirs(stats_dir, exist_ok=True)
                            
                            with open(stats_dir / f"{file_path.name}_stats.txt", 'w') as f:
                                f.write(f"Line count: {line_count}\n")
                            processed += 1
                    
                    except Exception as e:
                        logger.error(f"Error processing file {file_path}: {e}")
            
            return f"Processed {processed} files with {conversion_type} conversion"
        
        else:
            raise ToolExecutionError(f"Unsupported processing type: {processing_type}")
    
    def _handle_backup(self, parameters):
        """Handle backup operations"""
        source_dir = parameters.get("source_directory")
        backup_dir = parameters.get("backup_directory", "backups")
        
        if not source_dir:
            raise ToolExecutionError("Source directory must be specified for backup")
        
        source_path = Path(source_dir)
        if not source_path.exists():
            raise ToolExecutionError(f"Source directory does not exist: {source_dir}")
        
        # Create backup directory if it doesn't exist
        backup_path = Path(backup_dir)
        os.makedirs(backup_path, exist_ok=True)
        
        # Create timestamped backup folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = parameters.get("backup_name", source_path.name)
        backup_folder = backup_path / f"{backup_name}_{timestamp}"
        os.makedirs(backup_folder, exist_ok=True)
        
        # Perform backup based on backup_type
        backup_type = parameters.get("backup_type", "full")
        
        if backup_type == "full":
            # Use system commands for copying
            if os.name == "nt":  # Windows
                cmd = f'xcopy "{source_path}" "{backup_folder}" /E /H /C /I'
            else:  # Linux/Mac
                cmd = f'cp -r "{source_path}"/* "{backup_folder}"/'
            
            subprocess.run(cmd, shell=True, check=True)
            
            # Count files backed up
            file_count = sum(1 for _ in backup_folder.glob("**/*") if _.is_file())
            return f"Full backup created at {backup_folder} ({file_count} files)"
        
        elif backup_type == "incremental":
            # For incremental, we need the last backup timestamp
            previous_backups = sorted([d for d in backup_path.glob(f"{backup_name}_*") if d.is_dir()])
            
            if not previous_backups:
                # If no previous backup, do a full backup
                if os.name == "nt":  # Windows
                    cmd = f'xcopy "{source_path}" "{backup_folder}" /E /H /C /I'
                else:  # Linux/Mac
                    cmd = f'cp -r "{source_path}"/* "{backup_folder}"/'
                
                subprocess.run(cmd, shell=True, check=True)
                
                file_count = sum(1 for _ in backup_folder.glob("**/*") if _.is_file())
                return f"Initial backup created at {backup_folder} ({file_count} files)"
            
            # Get last backup time from folder name
            last_backup = previous_backups[-1]
            last_backup_time_str = last_backup.name.split("_")[-2] + "_" + last_backup.name.split("_")[-1]
            
            try:
                last_backup_time = datetime.strptime(last_backup_time_str, "%Y%m%d_%H%M%S")
                # Convert to timestamp
                last_backup_timestamp = last_backup_time.timestamp()
            except ValueError:
                # If we can't parse the time, default to 24 hours ago
                last_backup_timestamp = time.time() - 86400
            
            # Copy only newer files
            copied_count = 0
            for file_path in source_path.glob("**/*"):
                if file_path.is_file():
                    # Check if file is newer than last backup
                    mod_time = file_path.stat().st_mtime
                    if mod_time > last_backup_timestamp:
                        # Calculate relative path
                        rel_path = file_path.relative_to(source_path)
                        target_path = backup_folder / rel_path
                        
                        # Create parent directories
                        os.makedirs(target_path.parent, exist_ok=True)
                        
                        # Copy file
                        with open(file_path, "rb") as src, open(target_path, "wb") as dst:
                            dst.write(src.read())
                        
                        copied_count += 1
            
            if copied_count == 0:
                # Remove empty backup directory if no files were copied
                os.rmdir(backup_folder)
                return "No new or modified files to backup"
            
            return f"Incremental backup created at {backup_folder} ({copied_count} files)"
        
        else:
            raise ToolExecutionError(f"Unsupported backup type: {backup_type}")
    
    def _schedule_task(self, task_id, task_type, task_parameters, schedule):
        """Schedule a task for later execution"""
        # Create tasks directory if it doesn't exist
        tasks_dir = Path("scheduled_tasks")
        os.makedirs(tasks_dir, exist_ok=True)
        
        # Calculate next run time
        now = datetime.now()
        
        if schedule == "hourly":
            next_run = now.replace(minute=0, second=0, microsecond=0)
            # Move to next hour if current time is past the hour mark
            if now.minute > 0 or now.second > 0:
                next_run = next_run.replace(hour=now.hour + 1)
        elif schedule == "daily":
            next_run = now.replace(hour=0, minute=0, second=0, microsecond=0)
            # Move to next day if current time is past midnight
            if now.hour > 0 or now.minute > 0 or now.second > 0:
                # Add one day
                next_run = next_run.replace(day=now.day + 1)
        elif schedule == "weekly":
            # Schedule for next Monday at midnight
            days_until_monday = (7 - now.weekday()) % 7
            if days_until_monday == 0 and (now.hour > 0 or now.minute > 0 or now.second > 0):
                days_until_monday = 7  # If today is Monday and time > 00:00:00, schedule for next Monday
            
            # Add days until next Monday
            next_run = now.replace(hour=0, minute=0, second=0, microsecond=0)
            # Add days_until_monday days
            next_run = next_run.replace(day=now.day + days_until_monday)
        else:
            raise ToolExecutionError(f"Unsupported schedule: {schedule}")
        
        # Create task file
        task_data = {
            "task_id": task_id,
            "task_type": task_type,
            "task_parameters": task_parameters,
            "schedule": schedule,
            "next_run": next_run.isoformat(),
            "created_at": now.isoformat(),
            "status": "scheduled"
        }
        
        task_file = tasks_dir / f"{task_id}.json"
        with open(task_file, "w") as f:
            json.dump(task_data, f, indent=2)
        
        return f"Task {task_id} scheduled for {next_run.isoformat()} ({schedule})"
    
    def _record_task_event(self, task_id, status, details):
        """Record task events to history file"""
        event_data = {
            "task_id": task_id,
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "details": details
        }
        
        history_file = self.task_history_dir / f"{task_id}_{status}.json"
        with open(history_file, "w") as f:
            json.dump(event_data, f, indent=2)
        
        # Also append to combined log
        log_file = self.task_history_dir / "task_history.log"
        with open(log_file, "a") as f:
            f.write(f"{datetime.now().isoformat()} - Task {task_id} {status}\n")