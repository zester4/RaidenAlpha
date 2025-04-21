import smtplib
import logging
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import os
from pathlib import Path
from .base_tool import Tool, ToolExecutionError

logger = logging.getLogger("gemini_agent")

class EmailIntegrationTool(Tool):
    def __init__(self):
        super().__init__(
            name="send_email",
            description="Send emails with attachments via SMTP",
            parameters={
                "type": "object",
                "properties": {
                    "recipient": {"type": "string", "description": "Email address of the recipient"},
                    "subject": {"type": "string", "description": "Email subject"},
                    "body": {"type": "string", "description": "Email body in HTML format"},
                    "attachments": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of file paths to attach"
                    }
                },
                "required": ["recipient", "subject", "body"]
            }
        )
        self.smtp_server = os.getenv("SMTP_SERVER")
        self.smtp_port = int(os.getenv("SMTP_PORT", 587))
        self.email_user = os.getenv("EMAIL_USER")
        self.email_password = os.getenv("EMAIL_PASSWORD")
        
        if not all([self.smtp_server, self.email_user, self.email_password]):
            raise ToolExecutionError("Email configuration missing from environment variables")

    def execute(self, **kwargs):
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_user
            msg['To'] = kwargs['recipient']
            msg['Subject'] = kwargs['subject']
            
            # Attach body
            msg.attach(MIMEText(kwargs['body'], 'html'))
            
            # Process attachments
            for file_path in kwargs.get('attachments', []):
                attachment = Path(file_path)
                if not attachment.is_file():
                    logger.warning(f"Attachment not found: {file_path}")
                    continue
                
                part = MIMEBase('application', 'octet-stream')
                with open(attachment, 'rb') as file:
                    part.set_payload(file.read())
                encoders.encode_base64(part)
                part.add_header(
                    'Content-Disposition',
                    f'attachment; filename="{attachment.name}"'
                )
                msg.attach(part)

            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.email_user, self.email_password)
                server.sendmail(self.email_user, kwargs['recipient'], msg.as_string())
            
            logger.info(f"Email sent to {kwargs['recipient']} with subject: {kwargs['subject']}")
            return {"status": "success", "message": "Email sent successfully"}

        except Exception as e:
            logger.error(f"Email sending failed: {str(e)}")
            raise ToolExecutionError(f"Failed to send email: {str(e)}")