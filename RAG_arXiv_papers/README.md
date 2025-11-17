
# Start the fastAPI app:

uvicorn main:app --reload

You can either test the end point by
curl http://127.0.0.1:8000/api/search/?q=xxx

or go to http://127.0.0.1:8000 and type your question and click Submit button


# Test the util functions and generate output files: report, json, and index file

python test.py