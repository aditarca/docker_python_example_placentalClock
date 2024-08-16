
# Define the base of your image.
# We recommend to always specify a tag. `latest` may not 
# always ensure reproducibility.
FROM python:3.11-slim


# Copy files over to the image.
# We recommend copying over each file individually, as to take
# advantage of cache building (which helps reduce build time).
COPY requirements.txt /usr/local/bin/

# Install needed libraries/packages.
# Your model will be run without network access, so the dependencies
# must be installed here (and not during code execution).
RUN pip install -r /usr/local/bin/requirements.txt

# Copy the necessary Python scripts and model
COPY run_model.py /usr/local/bin/
COPY model_test_SC1.pkl /usr/local/bin/

# Make the script executable
RUN chmod a+x /usr/local/bin/run_model.py


# Set the main command of the image.
# We recommend using this form instead of `ENTRYPOINT command param1`.
ENTRYPOINT ["python", "/usr/local/bin/run_model.py"]


