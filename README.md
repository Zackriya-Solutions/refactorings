# @bpmn-io/refactorings 👷

Refactorings for bpmn-js, powered by AI!

## Setup

```bash
npm install
npm start
```

## Running suggestions backend

### Running with Docker
1. Go to the `suggestion_backend` folder:  
   `cd suggestion_backend`
2. Build and Run the app:  
   `docker compose up`
   Access at `http://localhost:8000`.

### Running Locally
1. Go to the `suggestion_backend` folder:  
   `cd suggestion_backend`
2. Install dependencies:  
   `pip install -r requirements.txt`
3. Ensure `all_elements.json` is in the `suggestion_backend` folder.
4. Start the app:  
   `uvicorn app:app --host 0.0.0.0 --port 8000`  
   Access at `http://localhost:8000`.

## License

MIT