# macOS v0.8.1 archive

The user requested a Mac package for archiving the same tested application after
the Windows handoff. Keep application behavior/version at 0.8.1 and retain the
already delivered Windows ZIP unchanged.

- Target Apple Silicon, matching this machine, with macOS 14 or later as required
  by the selected NumPy/SciPy wheels. Do not imply Intel/universal support.
- Use a verified CPython 3.11.15 standalone archive and hash-locked Mac wheels,
  not a copy of the developer virtualenv. Include the verified hand model.
- Deliver a ZIP with executable Start/Demo/Diagnose `.command` entries, Chinese
  instructions, model/dependency notices, source commit and file checksums.
- Reuse the existing portable bootstrap, making only platform-specific launcher
  labels and isolated Python invocation consistent across Mac/Windows.
- Validate the final archive by extracting into a different path (including spaces
  and Chinese characters), running its own Python/diagnostics and desktop demo,
  and confirming all dependency imports resolve inside the extracted runtime.
- Do not open the camera, send mouse input, change permissions or restart the user's
  active instance. Explain the usual Mac camera/accessibility/input-monitoring
  permissions in the guide. A signed/notarized app installer is outside this archive.

The archive includes application code, pinned dependencies, runtime/model provenance
and the portable build scripts so the archived version can be reconstructed.
