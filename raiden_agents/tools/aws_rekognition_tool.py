# Code for raiden_agents/tools/aws_rekognition_tool.py
import logging
import json
import traceback
import base64
import os
from pathlib import Path # Used for local file handling
# Assuming boto3 is installed in the environment
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError # Import specific exceptions
from datetime import datetime
from raiden_agents.tools.base_tool import Tool, ToolExecutionError # Import base Tool and exceptions

logger = logging.getLogger("gemini_agent") # Assuming logger is configured elsewhere

# Assuming AWS credentials are accessible
try:
    # Prefer explicit environment variables as per .env.example
    aws_access_key = os.environ.get("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    # AWS region is also needed
    aws_region = os.environ.get("AWS_REGION", "us-east-1") # Default to us-east-1 if not set
except Exception as e:
    logger.error(f"Could not access AWS environment variables in aws_rekognition_tool.py: {e}")
    aws_access_key = None
    aws_secret_key = None
    aws_region = "us-east-1" # Keep default

class AWSRekognitionTool(Tool):
    def __init__(self):
        super().__init__(
            name="aws_rekognition",
            description="Perform facial recognition and image analysis using AWS Rekognition",
            parameters={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["detect_faces", "compare_faces", "detect_labels", "detect_text"],
                        "description": "Type of Rekognition operation to perform"
                    },
                    "source_image": {
                        "type": "string",
                        "description": "Path to source image file or base64-encoded image data URL (e.g., data:image/jpeg;base64,...)"
                    },
                    "target_image": {
                        "type": "string",
                        "description": "Path to target image file or base64-encoded image data URL for face comparison (only for compare_faces)"
                    },
                    "similarity_threshold": {
                        "type": "number",
                        "description": "Minimum similarity threshold for face comparison (0-100). Default is 80."
                    }
                },
                "required": ["operation", "source_image"]
            }
        )

    def _get_image_bytes(self, image_identifier):
        """Helper to read image file or decode base64."""
        if image_identifier.startswith('data:image'):
            # Handle base64 encoded image data URL
            try:
                header, encoded = image_identifier.split(',', 1)
                # Optional: check header for mime type if needed
                # mime_type = header.split(':')[1].split(';')[0]
                image_bytes = base64.b64decode(encoded)
                logger.debug(f"Decoded base64 image data (length: {len(image_bytes)} bytes).")
                return image_bytes
            except Exception as e:
                raise ToolExecutionError(f"Failed to decode base64 image data: {e}")
        else:
            # Handle file path
            image_path = Path(image_identifier)
            if not image_path.is_file():
                 raise ToolExecutionError(f"Image file not found: {image_identifier}")
            try:
                with open(image_path, 'rb') as f:
                    image_bytes = f.read()
                logger.debug(f"Read image file '{image_identifier}' (length: {len(image_bytes)} bytes).")
                return image_bytes
            except Exception as e:
                raise ToolExecutionError(f"Failed to read image file '{image_identifier}': {e}")


    def execute(self, **kwargs):
        self.validate_args(kwargs)

        operation = kwargs.get("operation")
        source_image_id = kwargs.get("source_image")
        target_image_id = kwargs.get("target_image") # Only for compare_faces
        similarity_threshold = kwargs.get("similarity_threshold", 80) # Default threshold

        logger.info(f"Executing AWS Rekognition operation: '{operation}'")

        if not aws_access_key or not aws_secret_key:
            raise ToolExecutionError("AWS credentials missing. Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables.")

        try:
            # Initialize Rekognition client with credentials and region
            rekognition = boto3.client(
                'rekognition',
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                region_name=aws_region
            )

            # Process source image identifier
            source_bytes = self._get_image_bytes(source_image_id)


            if operation == "detect_faces":
                response = rekognition.detect_faces(
                    Image={'Bytes': source_bytes},
                    Attributes=['ALL'] # Request all attributes (age, gender, emotions, etc.)
                )
                return self._format_face_detection(response)

            elif operation == "compare_faces":
                if not target_image_id:
                    raise ToolExecutionError("'target_image' (path or base64) required for compare_faces operation.")

                # Process target image identifier
                target_bytes = self._get_image_bytes(target_image_id)

                response = rekognition.compare_faces(
                    SourceImage={'Bytes': source_bytes},
                    TargetImage={'Bytes': target_bytes},
                    SimilarityThreshold=similarity_threshold # Use provided or default threshold
                )
                return self._format_face_comparison(response)

            elif operation == "detect_labels":
                response = rekognition.detect_labels(
                    Image={'Bytes': source_bytes},
                    MaxLabels=10, # Limit the number of labels
                    MinConfidence=70 # Only return labels with confidence >= 70%
                )
                return self._format_label_detection(response)

            elif operation == "detect_text":
                response = rekognition.detect_text(
                    Image={'Bytes': source_bytes}
                )
                return self._format_text_detection(response)

            else:
                 # Should be caught by validate_args enum, but as a fallback
                 raise ToolExecutionError(f"Unsupported Rekognition operation: {operation}")


        except (NoCredentialsError, PartialCredentialsError):
            logger.error("AWS credentials not found or incomplete.")
            raise ToolExecutionError("AWS credentials not found or incomplete. Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY.")
        except ClientError as e:
            # Catch AWS SDK client errors (e.g., InvalidParameterException, ImageTooLargeException)
            logger.error(f"AWS Rekognition Client Error: {e.response['Error']['Code']} - {e.response['Error']['Message']}")
            raise ToolExecutionError(f"AWS Rekognition API error: {e.response['Error']['Message']}")
        except ToolExecutionError as e:
             # Re-raise our own tool execution errors (like file not found, invalid base64)
             raise e
        except Exception as e:
            logger.error(f"Unexpected AWS Rekognition error: {e}", exc_info=True)
            traceback.print_exc() # Print traceback to logs
            raise ToolExecutionError(f"Failed to perform AWS Rekognition operation: {e}")

    def _format_face_detection(self, response):
        faces = response.get('FaceDetails', [])
        results = []
        for i, face in enumerate(faces):
            face_info = {
                'index': i, # Add index for easier reference
                'confidence': face.get('Confidence'),
                'age_range': face.get('AgeRange'),
                'gender': face.get('Gender', {}).get('Value'),
                'emotions': [
                    {'type': e.get('Type'), 'confidence': e.get('Confidence')}
                    for e in face.get('Emotions', [])
                ],
                'pose': face.get('Pose'),
                'landmarks': len(face.get('Landmarks', [])), # Just count landmarks
                'quality': face.get('Quality'),
                'smile': face.get('Smile', {}).get('Value'),
                'sunglasses': face.get('Sunglasses', {}).get('Value'),
                'eyeglasses': face.get('Eyeglasses', {}).get('Value'),
                'beard': face.get('Beard', {}).get('Value'),
                'mustache': face.get('Mustache', {}).get('Value'),
                'eyes_open': face.get('EyesOpen', {}).get('Value'),
                'mouth_open': face.get('MouthOpen', {}).get('Value'),
            }
            # Clean up None values for cleaner output if needed
            cleaned_info = {k: v for k, v in face_info.items() if v is not None}
            results.append(cleaned_info)

        if not results:
             return "No faces detected in the image."

        return f"Detected {len(faces)} face(s):\n" + json.dumps(results, indent=2)

    def _format_face_comparison(self, response):
        matches = response.get('FaceMatches', [])
        unmatched_faces_source = response.get('UnmatchedFaces', [])
        unmatched_faces_target = response.get('SourceImageFace', []) # Source image can have one face, target can have many

        output = []
        if matches:
            match_summaries = [
                f"Similarity: {match['Similarity']:.1f}% (Face ID: {match.get('Face', {}).get('FaceId', 'N/A')})"
                for match in matches
            ]
            output.append(f"Found {len(matches)} matching face(s): " + ", ".join(match_summaries))
        else:
             output.append("No significant face matches found above the similarity threshold.")

        # You might also want to report unmatched faces or the face found in the source image
        if unmatched_faces_source:
            output.append(f"{len(unmatched_faces_source)} face(s) in the source image did not match.")
        if unmatched_faces_target and len(matches) == 0: # Only mention target faces if no matches were found at all
             output.append(f"{len(unmatched_faces_target)} face(s) found in the target image.")

        return "\n".join(output)


    def _format_label_detection(self, response):
        labels = response.get('Labels', [])
        if not labels:
             return "No labels detected in the image above the confidence threshold."

        # Sort labels by confidence
        sorted_labels = sorted(labels, key=lambda x: x.get('Confidence', 0), reverse=True)

        output = ["Detected labels:"]
        for label in sorted_labels:
            name = label.get('Name', 'N/A')
            confidence = label.get('Confidence', 0.0)
            instances = label.get('Instances', [])
            parents = [p.get('Name') for p in label.get('Parents', [])]

            label_info = f"- {name} ({confidence:.1f}%)"
            if parents:
                label_info += f" (Parents: {', '.join(parents)})"
            if instances:
                 label_info += f" ({len(instances)} instance{'s' if len(instances) > 1 else ''})"

            output.append(label_info)

        return "\n".join(output)


    def _format_text_detection(self, response):
        texts = response.get('TextDetections', [])
        if not texts:
             return "No text detected in the image."

        # Filter for lines or words and format
        detected_lines = [t.get('DetectedText') for t in texts if t.get('Type') == 'LINE' and t.get('Confidence', 0) > 80] # Only show lines with high confidence

        if not detected_lines:
             # If no high-confidence lines, show all text blocks (words/lines)
             detected_lines = [f"{t.get('DetectedText')} ({t.get('Type')}, {t.get('Confidence', 0):.1f}%)" for t in texts]
             if not detected_lines: return "No text detected in the image." # Still no text

        return "Detected text (Lines with >80% confidence or all detected text):\n" + "\n".join([f"- {line}" for line in detected_lines])
