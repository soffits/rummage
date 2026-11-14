# Queue

- Images and scanned PDFs in Seafile libraries are found by name only; OCR or image vectors for documents would need the image encoder on the server.
- Audio and video are not indexed.
- A message that another agent sends into a DSH session through `open_session` without the "[Sent by the agent" prefix is attributed to the web user.
- Vectors are loaded from SQLite on every search; a memory-mapped cache would help once an index passes a few hundred thousand segments.
