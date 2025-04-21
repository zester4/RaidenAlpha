import os
import logging
from datetime import datetime
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from .base_tool import Tool, ToolExecutionError

logger = logging.getLogger("gemini_agent")

class CalendarSchedulingTool(Tool):
    SCOPES = ['https://www.googleapis.com/auth/calendar']
    
    def __init__(self):
        super().__init__(
            name="schedule_event",
            description="Schedule Google Calendar events",
            parameters={
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "description": "Event title"},
                    "start_time": {"type": "string", "format": "date-time"},
                    "end_time": {"type": "string", "format": "date-time"},
                    "attendees": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of email addresses"
                    },
                    "location": {"type": "string"},
                    "description": {"type": "string"}
                },
                "required": ["summary", "start_time", "end_time"]
            }
        )
        self.credentials = self._authenticate()

    def _authenticate(self):
        creds = None
        token_path = os.path.join(os.path.expanduser("~"), ".config", "calendar_token.json")
        
        if os.path.exists(token_path):
            creds = Credentials.from_authorized_user_file(token_path, self.SCOPES)
        
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                raise ToolExecutionError("Calendar authentication required - run OAuth flow first")
        
        return creds

    def execute(self, **kwargs):
        try:
            service = build('calendar', 'v3', credentials=self.credentials)
            
            event = {
                'summary': kwargs['summary'],
                'location': kwargs.get('location', ''),
                'description': kwargs.get('description', ''),
                'start': {'dateTime': kwargs['start_time'], 'timeZone': 'UTC'},
                'end': {'dateTime': kwargs['end_time'], 'timeZone': 'UTC'},
                'attendees': [{'email': email} for email in kwargs.get('attendees', [])],
            }

            event = service.events().insert(
                calendarId='primary',
                body=event
            ).execute()
            
            logger.info(f"Created calendar event: {event.get('htmlLink')}")
            return {
                "status": "success",
                "event_id": event['id'],
                "link": event.get('htmlLink'),
                "start": event['start']['dateTime']
            }

        except Exception as e:
            logger.error(f"Calendar event creation failed: {str(e)}")
            raise ToolExecutionError(f"Failed to create calendar event: {str(e)}")