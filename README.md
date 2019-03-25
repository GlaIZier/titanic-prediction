To install global dependencies if they are needed:
```
sudo apt-get install python3.6-tk
```
In the project's directory:
```
python3 -m venv --system-site-packages ./venv
source venv/bin/activate
pip3 install -r requirements.txt
```
To deactivate:
```$xslt
deactivate
```

