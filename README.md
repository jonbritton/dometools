# dometools
Dome Tools:  Tools for domes, and such.  Utilities for esoteric production and playback tasks in immersive theaters/apps and the planetarium VFX pipeline.

```
bin/
|
|- make7th : converts EXR / PNG sequences to the proprietary 
|            7TH format, but this can be automated and 
|            distributed across a renderfarm. This was 
|            reverse-engineered, so: trust but verify.
|
|- make7th-job: a cron-friendly wrapper for the above.
|
|- notion_auth : session setup for automating updates to our Notion DB.
|
|- logrotator: rotates logs. 
|

ansible/
|
|- patch-compliance.yml : idempotent playbook for enforcing sys parity
|
|- deadline-client-setup.yml : sets up Deadline worker nodes. It 
|             is redundant with terraform setup from VizBurst, 
|             except that this includes the Docker containerization.
|

domecontrol/
|- pitch_server :  realtime pitch detection, facilitates the 
|            "roomful of singers control a full-dome game with 
|             their voice", theater experience.
|
|- color_detector : uses a cheap camera for the "colored light
|             player voting" via hue value of detected blobs
|
|- domecontrol_dashboard : flask app for testing the detectors
|
```

### Currently offline:

fixation:  monitors time and resources spent on each shot to determine where artists are just getting too knob-twiddly and what shots aren't getting enough attention.

vizfilespy: EXR metadata pushed to Notion — pulls embedded EXR header metadata (colorspace, camera info, frame range) for a shot and pushes it into the corresponding Notion entry.

A bunch of file format scripts that had too much cussing in the comments. 

All the Prometheus and Grafana stuff.

All the OpenStack stuff.

These will be added when I get a minute to format them more consistently.

