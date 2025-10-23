import requests
import json
import urllib3

# Disable SSL warnings when verify=False is used
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class ProgressReporter:
    def __init__(self, webhook_url=None):
        self.webhook_url = webhook_url

    def update(self, percent, message, job_id=None, video_id=None, clips=None, total_clips=None, error_message=None):
        if not self.webhook_url:
            print(f"📊 Progress: {percent}% - {message} (No webhook URL)")
            return
        
        # Laravel validator requires video_id as required integer
        if not video_id:
            print("⚠️ Warning: video_id is required for Laravel webhook")
            return
            
        payload = {
            "video_id": int(video_id),  # Required integer
            "progress": percent,        # Nullable integer 0-100
            "message": message          # Nullable string
        }
        
        # Set status based on progress and error state
        if error_message:
            payload["status"] = "failed"
            payload["error_message"] = error_message
        elif percent == 100:
            payload["status"] = "completed"
        else:
            payload["status"] = "processing"
        
        # Add clips array if provided (Laravel format)
        if clips is not None:
            payload["clips"] = clips
        
        try:
            print(f"Sending progress to webhook: {self.webhook_url}", flush=True)
            print(f"Payload: {json.dumps(payload, indent=2)}", flush=True)
            
            response = requests.post(self.webhook_url, json=payload, timeout=5, verify=False)
            
            print(f"Response Status: {response.status_code}", flush=True)
            print(f"Response Headers: {dict(response.headers)}", flush=True)
            # Skip response body to avoid encoding issues
            print("Response received (body skipped to avoid encoding issues)", flush=True)
            
            if response.status_code == 200:
                print(f"Progress update sent successfully: {percent}% - {message}")
            else:
                print(f"Progress update failed with status {response.status_code}: {response.text}")
                
        except requests.exceptions.Timeout:
            print(f"Webhook timeout after 5 seconds: {self.webhook_url}")
        except Exception as e:
            print(f"Unexpected error sending progress: {str(e)}")
