#! /bin/sh
echo "======================================================================"
echo "Welcome to to the setup. This will setup the local virtual env." 
echo "And then it will install all the required python libraries."
echo "You can rerun this without any issues."
echo "----------------------------------------------------------------------"
# Measure the time before installing Python libraries
start_time=$(date +%s)

if [ -d ".venv" ];
then
    echo ".venv folder exists. Installing using pip"
else
    echo "creating .venv and install using pip"
    python3 -m venv .venv
fi

# Activate virtual env
echo "Activating virtual environment"
. .venv/bin/activate

# Upgrade the PIP
#echo "Upgrading PIP..."
#pip install --upgrade pip
#echo "Installing Python libraries from requirements.txt..."
pip install Flask
pip install -r requirements.txt

# Print installed packages
echo "======================================================================"
echo "Installed Python packages:"
pip freeze
echo "======================================================================"

end_time=$(date +%s)

# Calculate and display the elapsed time
elapsed_time=$((end_time - start_time))
echo "Installation completed in $elapsed_time seconds."
#run the app
python3 main.py
# Work done. so deactivate the virtual env
deactivate
