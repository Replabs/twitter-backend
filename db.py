"""Firebase database wrapper."""

from firebase_admin import firestore, credentials, initialize_app

# Prepare the DB.
cred = credentials.Certificate("./firebase_service_account_key.json")
app = initialize_app(cred)
db = firestore.client()
