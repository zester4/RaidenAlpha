import logging
import base64
import json
import tempfile
from dotenv import load_dotenv
import os
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
from github import Github, GithubException
from .base_tool import Tool, ToolExecutionError, GitHubToolError # Import base and specific error


load_dotenv()
# Assume logger and vector_db are configured and accessible in the execution environment (app.py)
logger = logging.getLogger("gemini_agent")

# Assume github_api_key is loaded from environment in app.py and accessible
# We might need a way to pass it during initialization or execution if not globally available
# For now, assume it's accessible via os.environ like other keys if needed directly here,
# but the provided execute method seems to expect it to be checked before calling.
# Let's add the necessary os import just in case, though the execute logic relies on app.py's check.
import os
github_api_key = os.environ.get("GITHUB_API_KEY") # Get key for check within execute

# Import vector_db from the main application context if needed for logging within the tool
# This is tricky; ideally, logging/DB interaction happens outside the tool's core logic
# or the DB connection is passed in. We'll rely on app.py's context for now.
try:
    from __main__ import vector_db
except ImportError:
    vector_db = None # Fallback if running standalone or vector_db isn't in __main__

class GitHubTool(Tool):
    def __init__(self):
        super().__init__(
            name="github_operations",
            description="Manage GitHub repositories, files and operations.",
            parameters={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": [
                            "list_repos",
                            "create_repo",
                            "read_file",
                            "write_file",
                            "list_files",
                            "clone_repo",
                            "create_directory",
                            "delete_file",
                            "delete_repo",
                            "create_branch",
                            "create_pull_request",
                            "merge_pull_request",
                            "list_branches",
                            "list_pull_requests",
                            "add_collaborator",
                            "get_commit_history",
                            "create_issue",
                            "list_issues",
                            "comment_on_issue",
                            "close_issue",
                            "get_repo_info",
                            "fork_repo"
                        ],
                        "description": "GitHub operation."
                    },
                    "repo_name": {"type": "string", "description": "[owner/]repo name."},
                    "file_path": {"type": "string", "description": "Path to file."},
                    "file_content": {"type": "string", "description": "Content to write."},
                    "description": {"type": "string", "description": "Repo/PR/Issue description."},
                    "private": {"type": "boolean", "description": "Make repo private (default: false)."},
                    "branch": {"type": "string", "description": "Branch name (default: main/master)."},
                    "commit_message": {"type": "string", "description": "Commit message."},
                    "path": {"type": "string", "description": "Directory path."},
                    "target_branch": {"type": "string", "description": "Target branch for PR."},
                    "base_branch": {"type": "string", "description": "Base branch for PR or new branch."},
                    "title": {"type": "string", "description": "Title for PR/Issue."},
                    "body": {"type": "string", "description": "Body content for PR/Issue/Comment."},
                    "collaborator": {"type": "string", "description": "Username to add as collaborator."},
                    "permission": {"type": "string", "description": "Permission level for collaborator (pull/push/admin)."},
                    "state": {"type": "string", "description": "State for issue/PR (open/closed)."},
                    "labels": {"type": "array", "items": {"type": "string"}, "description": "Labels for issue/PR."},
                    "assignees": {"type": "array", "items": {"type": "string"}, "description": "Assignees for issue/PR."}
                },
                "required": ["operation"]
            },
            required=["operation"]
        )

    def _get_repo(self, github, repo_name):
        if not repo_name:
            raise GitHubToolError("'repo_name' required.")
        try:
            return github.get_repo(repo_name)
        except GithubException as e:
            if e.status == 404:
                raise GitHubToolError(f"Repo '{repo_name}' not found/accessible.")
            else:
                raise GitHubToolError(f"Error accessing repo '{repo_name}': {e.status} - {e.data.get('message', str(e))}")
        except Exception as e:
            logger.error(f"Unexpected error getting repo '{repo_name}': {str(e)}", exc_info=True)
            raise GitHubToolError(f"Error getting repo '{repo_name}': {str(e)}")

    def execute(self, **kwargs):
        self.validate_args(kwargs)
        operation = kwargs.get("operation")
        logger.info(f"Executing GitHub op: {operation}")

        # Check for API key (assuming it's loaded in the main app environment)
        if not github_api_key:
            raise GitHubToolError("GitHub API key missing.")

        try:
            g = Github(github_api_key)
            user = g.get_user() # Get authenticated user

            if operation == "list_repos":
                repos = list(user.get_repos(affiliation='owner'))
                if not repos:
                    return "No owned repositories found."
                repo_list = [f"- {r.full_name} ({'private' if r.private else 'public'})" for r in repos[:30]] # Limit output
                return f"Your repositories ({min(len(repos), 30)} shown):\n" + "\n".join(repo_list)

            elif operation == "create_repo":
                repo_name = kwargs.get("repo_name")
                if not repo_name or "/" in repo_name: # Basic validation
                    raise GitHubToolError("Valid repository name (without owner/slash) required for creation.")
                desc = kwargs.get("description", "")
                priv = kwargs.get("private", False)
                logger.info(f"Creating repo: {repo_name}")
                repo = user.create_repo(name=repo_name, description=desc, private=priv, auto_init=True) # auto_init to avoid empty repo issues
                # Log to vector DB if available
                if vector_db and vector_db.is_ready():
                    vector_db.add(
                        f"Created GitHub repo: {repo.full_name}",
                        {"type": "github_action", "action": "create_repo", "repo": repo.full_name, "time": datetime.now().isoformat()}
                    )
                return f"Repository '{repo.full_name}' created successfully: {repo.html_url}"

            # Operations requiring repo_name need it validated first
            repo_name = kwargs.get("repo_name")
            if not repo_name and operation not in ["list_repos", "create_repo"]:
                 raise GitHubToolError(f"'repo_name' is required for operation '{operation}'.")

            repo = self._get_repo(g, repo_name) # Get repo object for subsequent operations

            if operation == "read_file":
                fp = kwargs.get("file_path")
                br = kwargs.get("branch", repo.default_branch)
                if not fp:
                    raise GitHubToolError("'file_path' required for read_file.")
                logger.info(f"Reading '{fp}' from '{repo.full_name}' branch '{br}'")
                try:
                    cf = repo.get_contents(fp, ref=br)
                    # Handle potential decoding errors
                    try:
                        content = base64.b64decode(cf.content).decode('utf-8')
                        return f"Content of '{fp}' in '{repo.full_name}' (branch: {br}):\n```\n{content}\n```"
                    except UnicodeDecodeError:
                         logger.warning(f"Could not decode file '{fp}' as UTF-8. Returning raw base64.")
                         return f"Content of '{fp}' (non-UTF8) in '{repo.full_name}' (branch: {br}):\nBase64: {cf.content}"
                except GithubException as e:
                    if e.status == 404:
                        raise GitHubToolError(f"File '{fp}' not found in branch '{br}'.")
                    else:
                        raise # Re-raise other GitHub errors

            elif operation == "write_file":
                fp = kwargs.get("file_path")
                fc = kwargs.get("file_content") # Content should be string
                cm = kwargs.get("commit_message")
                br = kwargs.get("branch", repo.default_branch)
                if not fp or fc is None or not cm:
                    raise GitHubToolError("'file_path', 'file_content', and 'commit_message' are required for write_file.")
                logger.info(f"Writing to '{fp}' in '{repo.full_name}' branch '{br}'")
                try:
                    # Check if file exists to update or create
                    contents = repo.get_contents(fp, ref=br)
                    commit = repo.update_file(contents.path, cm, fc, contents.sha, branch=br)
                    action = "updated"
                except GithubException as e:
                    if e.status == 404: # File not found, create it
                        commit = repo.create_file(fp, cm, fc, branch=br)
                        action = "created"
                    else:
                        raise # Re-raise other GitHub errors
                # Log to vector DB if available
                if vector_db and vector_db.is_ready():
                    vector_db.add(
                        f"GitHub file {action}: {repo.full_name}/{fp}",
                        {"type": "github_action", "action": "write_file", "repo": repo.full_name, "path": fp, "time": datetime.now().isoformat()}
                    )
                return f"File '{fp}' {action} in '{repo.full_name}' (branch: {br}). Commit SHA: {commit['commit'].sha}"

            elif operation == "list_files":
                path = kwargs.get("path", "") # List root if path is empty
                br = kwargs.get("branch", repo.default_branch)
                logger.info(f"Listing files in '{repo.full_name}/{path}' branch '{br}'")
                try:
                    contents = repo.get_contents(path, ref=br)
                    if not contents:
                        return f"Directory '{path}' in branch '{br}' is empty or does not exist."
                    # Ensure contents is a list (it is for directories)
                    if not isinstance(contents, list):
                         contents = [contents] # Wrap single file object in list
                    file_list = [f"- {'[DIR] ' if item.type == 'dir' else ''}{item.path}" for item in contents]
                    return f"Files/Directories in '{repo.full_name}/{path}' (branch: {br}):\n" + "\n".join(file_list)
                except GithubException as e:
                     if e.status == 404:
                          raise GitHubToolError(f"Path '{path}' not found in branch '{br}'.")
                     else:
                          raise

            elif operation == "clone_repo":
                 # Note: Cloning large repos might fail due to timeout or disk space.
                 # This operation clones to a temporary directory which is then removed.
                 # Consider security implications if running in a shared environment.
                clone_dir = Path(tempfile.mkdtemp())
                repo_url = repo.clone_url # Use HTTPS clone URL
                logger.info(f"Attempting to clone '{repo.full_name}' to temporary directory {clone_dir}")
                try:
                    # Use subprocess to run git clone
                    process = subprocess.run(
                        ['git', 'clone', '--depth', '1', repo_url, str(clone_dir)], # Shallow clone
                        capture_output=True, text=True, check=True, timeout=120 # 2 min timeout
                    )
                    logger.info(f"Clone successful (stdout): {process.stdout}")
                    result = f"Repository '{repo.full_name}' was temporarily cloned successfully (and then removed)."
                except subprocess.CalledProcessError as e:
                    logger.error(f"Git clone command failed. Return code: {e.returncode}\nStderr: {e.stderr}")
                    raise GitHubToolError(f"Git clone failed: {e.stderr}")
                except subprocess.TimeoutExpired:
                    logger.error("Git clone operation timed out after 120 seconds.")
                    raise GitHubToolError("Git clone operation timed out.")
                except FileNotFoundError:
                    logger.error("Git command not found. Ensure git is installed and in PATH.")
                    raise GitHubToolError("Git command not found on the system.")
                except Exception as e:
                    logger.error(f"An unexpected error occurred during cloning: {str(e)}", exc_info=True)
                    raise GitHubToolError(f"Cloning error: {str(e)}")
                finally:
                    # Ensure temporary directory is always removed
                    if clone_dir.exists():
                        try:
                            shutil.rmtree(clone_dir)
                            logger.info(f"Removed temporary clone directory: {clone_dir}")
                        except Exception as e_rm:
                            logger.error(f"Failed to remove temporary directory {clone_dir}: {e_rm}")
                return result # Return success message if no exceptions were raised

            elif operation == "create_directory":
                path = kwargs.get("path")
                branch = kwargs.get("branch", repo.default_branch)
                commit_message = kwargs.get("commit_message", f"Create directory {path} via agent")
                if not path:
                    raise GitHubToolError("'path' is required for create_directory.")

                # GitHub API requires creating a file (like .gitkeep) to represent a directory
                dir_path = path.strip('/') # Remove leading/trailing slashes
                file_path = f"{dir_path}/.gitkeep" # Standard practice file
                logger.info(f"Attempting to create directory '{dir_path}' via file '{file_path}' in branch '{branch}'")

                try:
                    # Check if the .gitkeep file already exists (implies directory exists)
                    repo.get_contents(file_path, ref=branch)
                    return f"Directory '{dir_path}' likely already exists in branch '{branch}' (found .gitkeep)."
                except GithubException as e:
                    if e.status == 404:
                        # File not found, proceed to create it
                        try:
                            repo.create_file(file_path, commit_message, "", branch=branch) # Empty content for .gitkeep
                            # Log to vector DB if available
                            if vector_db and vector_db.is_ready():
                                vector_db.add(
                                    f"Created directory: {dir_path} in {repo.full_name}",
                                    {"type": "github_action", "action": "create_directory", "repo": repo.full_name, "path": dir_path, "time": datetime.now().isoformat()}
                                )
                            return f"Directory '{dir_path}' created successfully in branch '{branch}' (via .gitkeep)."
                        except GithubException as create_e:
                             logger.error(f"Failed to create file '{file_path}': {create_e.status} - {create_e.data.get('message', str(create_e))}")
                             raise GitHubToolError(f"Failed to create directory file: {create_e.data.get('message', str(create_e))}")
                    else:
                        # Other error getting contents (permissions?)
                        raise GitHubToolError(f"Error checking directory existence: {e.status} - {e.data.get('message', str(e))}")

            elif operation == "delete_file":
                fp = kwargs.get("file_path")
                cm = kwargs.get("commit_message", f"Delete file {fp} via agent")
                br = kwargs.get("branch", repo.default_branch)
                if not fp:
                    raise GitHubToolError("'file_path' is required for delete_file.")
                logger.info(f"Attempting to delete '{fp}' from '{repo.full_name}' branch '{br}'")
                try:
                    contents = repo.get_contents(fp, ref=br) # Need SHA to delete
                    repo.delete_file(contents.path, cm, contents.sha, branch=br)
                    # Log to vector DB if available
                    if vector_db and vector_db.is_ready():
                        vector_db.add(
                            f"Deleted file: {fp} from {repo.full_name}",
                            {"type": "github_action", "action": "delete_file", "repo": repo.full_name, "path": fp, "time": datetime.now().isoformat()}
                        )
                    return f"File '{fp}' deleted successfully from branch '{br}'."
                except GithubException as e:
                    if e.status == 404:
                        raise GitHubToolError(f"File '{fp}' not found in branch '{br}'. Cannot delete.")
                    else:
                        raise # Re-raise other GitHub errors

            elif operation == "delete_repo":
                 # This is a destructive operation, ensure user confirmation if possible externally
                 logger.warning(f"Attempting to delete repository: {repo.full_name}")
                 repo.delete()
                 # Log to vector DB if available
                 if vector_db and vector_db.is_ready():
                     vector_db.add(
                         f"Deleted repository: {repo.full_name}",
                         {"type": "github_action", "action": "delete_repo", "repo": repo.full_name, "time": datetime.now().isoformat()}
                     )
                 return f"Repository '{repo.full_name}' deleted successfully."

            elif operation == "create_branch":
                new_branch = kwargs.get("branch")
                base_branch = kwargs.get("base_branch", repo.default_branch)
                if not new_branch:
                    raise GitHubToolError("'branch' (new branch name) is required.")
                logger.info(f"Creating branch '{new_branch}' from '{base_branch}' in '{repo.full_name}'")
                try:
                    source = repo.get_branch(base_branch)
                    repo.create_git_ref(ref=f"refs/heads/{new_branch}", sha=source.commit.sha)
                    # Log to vector DB if available
                    if vector_db and vector_db.is_ready():
                        vector_db.add(
                            f"Created branch: {new_branch} in {repo.full_name}",
                            {"type": "github_action", "action": "create_branch", "repo": repo.full_name, "branch": new_branch, "time": datetime.now().isoformat()}
                        )
                    return f"Branch '{new_branch}' created successfully from '{base_branch}'."
                except GithubException as e:
                    if e.status == 422: # Already exists
                        raise GitHubToolError(f"Branch '{new_branch}' already exists or base branch '{base_branch}' not found.")
                    else:
                        raise

            elif operation == "create_pull_request":
                title = kwargs.get("title")
                head = kwargs.get("branch") # The branch with changes
                base = kwargs.get("base_branch", repo.default_branch) # The branch to merge into
                body = kwargs.get("body", "")
                if not title or not head:
                    raise GitHubToolError("'title' and 'branch' (head branch) are required.")
                logger.info(f"Creating PR in '{repo.full_name}': '{title}' ({head} -> {base})")
                try:
                    pr = repo.create_pull(title=title, body=body, head=head, base=base)
                    # Log to vector DB if available
                    if vector_db and vector_db.is_ready():
                        vector_db.add(
                            f"Created PR #{pr.number}: {title} in {repo.full_name}",
                            {"type": "github_action", "action": "create_pull_request", "repo": repo.full_name, "pr_number": pr.number, "time": datetime.now().isoformat()}
                        )
                    return f"Pull Request #{pr.number} created successfully: {pr.html_url}"
                except GithubException as e:
                     # Common error: No commits between branches
                     if e.status == 422 and "No commits between" in e.data.get('message', ''):
                          raise GitHubToolError(f"Cannot create PR: No commits found between '{base}' and '{head}'.")
                     else:
                          raise

            elif operation == "merge_pull_request":
                pr_number_str = kwargs.get("number") # API expects number, not string
                commit_message = kwargs.get("commit_message", "") # Optional commit message
                if not pr_number_str:
                    raise GitHubToolError("'number' (Pull Request number) is required.")
                try:
                    pr_number = int(pr_number_str)
                except ValueError:
                     raise GitHubToolError("'number' must be an integer PR number.")

                logger.info(f"Attempting to merge PR #{pr_number} in '{repo.full_name}'")
                pr = repo.get_pull(pr_number)

                if pr.merged:
                    return f"Pull Request #{pr_number} is already merged."
                if not pr.mergeable:
                     # Provide more context if possible
                     merge_state = pr.mergeable_state
                     return f"Pull Request #{pr_number} cannot be merged automatically. State: {merge_state}. Manual review may be needed."

                try:
                    merge_status = pr.merge(commit_message=commit_message)
                    # Log to vector DB if available
                    if vector_db and vector_db.is_ready():
                        vector_db.add(
                            f"Merged PR #{pr_number} in {repo.full_name}",
                            {"type": "github_action", "action": "merge_pull_request", "repo": repo.full_name, "pr_number": pr_number, "time": datetime.now().isoformat()}
                        )
                    return f"Pull Request #{pr_number} merged successfully. Merge commit SHA: {merge_status.sha}"
                except GithubException as merge_e:
                     logger.error(f"Failed to merge PR #{pr_number}: {merge_e.status} - {merge_e.data.get('message', str(merge_e))}")
                     raise GitHubToolError(f"Failed to merge PR: {merge_e.data.get('message', str(merge_e))}")

            elif operation == "list_branches":
                logger.info(f"Listing branches for '{repo.full_name}'")
                branches = repo.get_branches()
                branch_list = [f"- {b.name}" for b in branches]
                if not branch_list: return f"No branches found in '{repo.full_name}'."
                return f"Branches in '{repo.full_name}':\n" + "\n".join(branch_list)

            elif operation == "list_pull_requests":
                state = kwargs.get("state", "open") # 'open', 'closed', or 'all'
                logger.info(f"Listing '{state}' pull requests for '{repo.full_name}'")
                prs = repo.get_pulls(state=state, sort='created', direction='desc')
                pr_list = [f"- #{pr.number}: {pr.title} (State: {pr.state}, User: {pr.user.login})" for pr in prs]
                if not pr_list: return f"No '{state}' pull requests found in '{repo.full_name}'."
                return f"Pull Requests in '{repo.full_name}' (State: {state}):\n" + "\n".join(pr_list)

            elif operation == "add_collaborator":
                username = kwargs.get("collaborator")
                permission = kwargs.get("permission", "push") # 'pull', 'push', 'admin', 'maintain', 'triage'
                if not username:
                    raise GitHubToolError("'collaborator' username is required.")
                valid_permissions = ['pull', 'push', 'admin', 'maintain', 'triage']
                if permission not in valid_permissions:
                     raise GitHubToolError(f"Invalid permission '{permission}'. Must be one of: {', '.join(valid_permissions)}")
                logger.info(f"Adding '{username}' as collaborator to '{repo.full_name}' with '{permission}' permission.")
                # Note: This sends an invitation, the user must accept it.
                repo.add_to_collaborators(username, permission=permission)
                # Log to vector DB if available
                if vector_db and vector_db.is_ready():
                    vector_db.add(
                        f"Invited collaborator {username} to {repo.full_name}",
                        {"type": "github_action", "action": "add_collaborator", "repo": repo.full_name, "collaborator": username, "time": datetime.now().isoformat()}
                    )
                return f"Invitation sent to '{username}' to collaborate on '{repo.full_name}' with '{permission}' permission."

            elif operation == "get_commit_history":
                fp = kwargs.get("file_path") # Optional: filter by file path
                br = kwargs.get("branch") # Optional: specify branch
                params = {}
                if fp: params['path'] = fp
                if br: params['sha'] = br # 'sha' can be branch name, tag, or commit SHA
                logger.info(f"Getting commit history for '{repo.full_name}' (Path: {fp or 'all'}, Branch: {br or 'default'})")
                commits = repo.get_commits(**params)
                # Limit output to recent commits
                commit_list = [f"- {c.sha[:7]}: {c.commit.message.splitlines()[0]} (by {c.commit.author.name} on {c.commit.author.date.strftime('%Y-%m-%d')})" for c in commits[:20]]
                scope = f"file '{fp}'" if fp else "repository"
                branch_info = f"branch '{br}'" if br else "default branch"
                if not commit_list: return f"No commit history found for {scope} on {branch_info}."
                return f"Recent commit history for {scope} in '{repo.full_name}' ({branch_info}, {len(commit_list)} shown):\n" + "\n".join(commit_list)

            elif operation == "create_issue":
                title = kwargs.get("title")
                body = kwargs.get("body", "")
                labels = kwargs.get("labels", []) # Should be list of strings
                assignees = kwargs.get("assignees", []) # Should be list of strings
                if not title:
                    raise GitHubToolError("'title' is required for create_issue.")
                logger.info(f"Creating issue in '{repo.full_name}': '{title}'")
                issue = repo.create_issue(title=title, body=body, labels=labels, assignees=assignees)
                # Log to vector DB if available
                if vector_db and vector_db.is_ready():
                    vector_db.add(
                        f"Created issue #{issue.number}: {title} in {repo.full_name}",
                        {"type": "github_action", "action": "create_issue", "repo": repo.full_name, "issue_number": issue.number, "time": datetime.now().isoformat()}
                    )
                return f"Issue #{issue.number} created successfully: {issue.html_url}"

            elif operation == "list_issues":
                state = kwargs.get("state", "open") # 'open', 'closed', 'all'
                logger.info(f"Listing '{state}' issues for '{repo.full_name}'")
                issues = repo.get_issues(state=state, sort='created', direction='desc')
                issue_list = [f"- #{issue.number}: {issue.title} (State: {issue.state}, User: {issue.user.login})" for issue in issues]
                if not issue_list: return f"No '{state}' issues found in '{repo.full_name}'."
                return f"Issues in '{repo.full_name}' (State: {state}):\n" + "\n".join(issue_list)

            elif operation == "comment_on_issue":
                issue_number_str = kwargs.get("number")
                body = kwargs.get("body")
                if not issue_number_str or not body:
                    raise GitHubToolError("'number' (Issue number) and 'body' are required.")
                try:
                    issue_number = int(issue_number_str)
                except ValueError:
                     raise GitHubToolError("'number' must be an integer issue number.")
                logger.info(f"Commenting on issue #{issue_number} in '{repo.full_name}'")
                issue = repo.get_issue(issue_number)
                comment = issue.create_comment(body)
                # Log to vector DB if available
                if vector_db and vector_db.is_ready():
                    vector_db.add(
                        f"Commented on issue #{issue_number} in {repo.full_name}",
                        {"type": "github_action", "action": "comment_on_issue", "repo": repo.full_name, "issue_number": issue_number, "time": datetime.now().isoformat()}
                    )
                return f"Comment added successfully to issue #{issue_number}: {comment.html_url}"

            elif operation == "close_issue":
                issue_number_str = kwargs.get("number")
                if not issue_number_str:
                    raise GitHubToolError("'number' (Issue number) is required.")
                try:
                    issue_number = int(issue_number_str)
                except ValueError:
                     raise GitHubToolError("'number' must be an integer issue number.")
                logger.info(f"Closing issue #{issue_number} in '{repo.full_name}'")
                issue = repo.get_issue(issue_number)
                if issue.state == 'closed':
                     return f"Issue #{issue_number} is already closed."
                issue.edit(state="closed")
                # Log to vector DB if available
                if vector_db and vector_db.is_ready():
                    vector_db.add(
                        f"Closed issue #{issue_number} in {repo.full_name}",
                        {"type": "github_action", "action": "close_issue", "repo": repo.full_name, "issue_number": issue_number, "time": datetime.now().isoformat()}
                    )
                return f"Issue #{issue_number} closed successfully."

            elif operation == "get_repo_info":
                 logger.info(f"Getting info for repository '{repo.full_name}'")
                 info = {
                     "name": repo.name,
                     "full_name": repo.full_name,
                     "description": repo.description or "N/A",
                     "url": repo.html_url,
                     "private": repo.private,
                     "fork": repo.fork,
                     "default_branch": repo.default_branch,
                     "stars": repo.stargazers_count,
                     "forks": repo.forks_count,
                     "watchers": repo.watchers_count,
                     "open_issues": repo.open_issues_count,
                     "language": repo.language or "N/A",
                     "created_at": repo.created_at.isoformat(),
                     "updated_at": repo.updated_at.isoformat(),
                     "size_kb": repo.size,
                 }
                 # Format nicely for output
                 info_str = "\n".join([f"- {key}: {value}" for key, value in info.items()])
                 return f"Repository Information for '{repo.full_name}':\n{info_str}"

            elif operation == "fork_repo":
                 logger.info(f"Attempting to fork repository '{repo.full_name}'")
                 # Forking creates the repo under the authenticated user's account
                 forked_repo = user.create_fork(repo)
                 # Log to vector DB if available
                 if vector_db and vector_db.is_ready():
                     vector_db.add(
                         f"Forked repository: {repo.full_name} to {forked_repo.full_name}",
                         {"type": "github_action", "action": "fork_repo", "source_repo": repo.full_name, "fork_repo": forked_repo.full_name, "time": datetime.now().isoformat()}
                     )
                 return f"Repository '{repo.full_name}' forked successfully to '{forked_repo.full_name}': {forked_repo.html_url}"

            else:
                # This case should ideally not be reached due to enum validation in schema
                raise GitHubToolError(f"Unsupported GitHub operation: '{operation}'")

        except GithubException as e:
            logger.error(f"GitHub API error during operation '{operation}': {e.status} - {e.data.get('message', str(e))}")
            raise GitHubToolError(f"GitHub API error: {e.status} - {e.data.get('message', str(e))}")
        except ToolExecutionError as e: # Catch specific tool errors (like missing args)
             raise e # Re-raise to be handled by the main loop
        except Exception as e:
            logger.error(f"Unexpected error during GitHub operation '{operation}': {str(e)}", exc_info=True)
            raise GitHubToolError(f"An unexpected error occurred: {str(e)}")
