#!/bin/bash

cd ~/actions-runner && sudo ./svc.sh stop && sudo ./svc.sh start
journalctl -u 'actions.runner*' -f     # wait for "Connected to GitHub" / "Listening for Jobs"
