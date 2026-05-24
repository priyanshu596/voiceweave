import requests
import time
import os
import sys

# Configuration
BASE_URL = os.getenv("API_BASE_URL", "http://localhost:5000")
SAMPLE_FILE = "sample_novel.txt"

def test_narrate():
    """Test the /api/narrate endpoint with a small JSON payload."""
    print("--- Testing /api/narrate ---")
    payload = {
        "text": "The mysterious traveler arrived at midnight. 'Who goes there?' the guard challenged."
    }
    
    try:
        response = requests.post(f"{BASE_URL}/api/narrate", json=payload)
        response.raise_for_status()
        data = response.json()
        print(f"Status Code: {response.status_code}")
        print(f"Response Status: {data.get('status')}")
        print(f"Audio URL: {data.get('audio_url')}")
        print("Blocks detected:")
        for block in data.get('blocks', []):
            print(f"  - [{block['speaker']} ({block['emotion']})]: {block['dialogue'][:50]}...")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error testing /api/narrate: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response content: {e.response.text}")
        return False

def test_upload_and_poll():
    """Test the /api/upload endpoint and poll for completion."""
    print("\n--- Testing /api/upload and Polling Status ---")
    
    if not os.path.exists(SAMPLE_FILE):
        print(f"Creating temporary sample file: {SAMPLE_FILE}")
        with open(SAMPLE_FILE, "w") as f:
            f.write("Chapter 1: The Beginning. Liam said, 'Hello Elena.' Elena sighed, 'Not you again.'")

    try:
        print(f"Uploading {SAMPLE_FILE}...")
        with open(SAMPLE_FILE, "rb") as f:
            files = {"file": (SAMPLE_FILE, f, "text/plain")}
            response = requests.post(f"{BASE_URL}/api/upload", files=files)
        
        response.raise_for_status()
        data = response.json()
        job_id = data.get("job_id")
        print(f"Upload successful. Job ID: {job_id}")
        
        # Polling
        print(f"Polling status for job {job_id}...")
        while True:
            status_response = requests.get(f"{BASE_URL}/api/status/{job_id}")
            status_response.raise_for_status()
            job_info = status_response.json()
            status = job_info.get("status")
            
            print(f"Current Status: {status}")
            
            if status == "completed":
                print("\nJob Completed Successfully!")
                print(f"Final Audio URL: {job_info.get('audio_url')}")
                return True
            elif status == "failed":
                print(f"\nJob Failed: {job_info.get('error')}")
                return False
            
            # Wait before next poll
            time.sleep(3)
            
    except requests.exceptions.RequestException as e:
        print(f"Error during upload/polling: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response content: {e.response.text}")
        return False

if __name__ == "__main__":
    print(f"Starting API tests against: {BASE_URL}")
    
    narrate_success = test_narrate()
    upload_success = test_upload_and_poll()
    
    print("\n" + "="*20)
    print("TEST SUMMARY")
    print(f"/api/narrate: {'PASSED' if narrate_success else 'FAILED'}")
    print(f"/api/upload & poll: {'PASSED' if upload_success else 'FAILED'}")
    print("="*20)
    
    if not (narrate_success and upload_success):
        sys.exit(1)
