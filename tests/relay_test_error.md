Traceback (most recent call last):
  File "tests/relay_test.py", line 26, in <module>
    main()
  File "tests/relay_test.py", line 10, in main
    GPIO.setup(RELAY_PIN, GPIO.OUT)  # Relay pin set as output
  File "/home/eskutcheon/.local/lib/python3.7/site-packages/Jetson/GPIO/gpio.py", line 333, in setup
    ch_infos = _channels_to_infos(channels, need_gpio=True)
  File "/home/eskutcheon/.local/lib/python3.7/site-packages/Jetson/GPIO/gpio.py", line 125, in _channels_to_infos
    for c in _make_iterable(channels)]
  File "/home/eskutcheon/.local/lib/python3.7/site-packages/Jetson/GPIO/gpio.py", line 125, in <listcomp>
    for c in _make_iterable(channels)]
  File "/home/eskutcheon/.local/lib/python3.7/site-packages/Jetson/GPIO/gpio.py", line 110, in _channel_to_info_lookup
    raise ValueError("Channel %s is invalid" % str(channel))
ValueError: Channel 10 is invalid
