import logging

# The HF Hub warns on every run that requests are unauthenticated (no HF_TOKEN). Our
# models are public, so the notice is noise; real errors still get through.
logging.getLogger("huggingface_hub.utils._http").setLevel(logging.ERROR)
