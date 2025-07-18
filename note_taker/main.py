import os
import pathlib
import re
import select
import sys
import tempfile
import termios
import time
import tty
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import openai
import scipy.io.wavfile
import sounddevice as sd
import typer
from rapidfuzz import fuzz, process
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.prompt import Confirm

app = typer.Typer()
console = Console()

TEMPLATE_CONTENT = """---
Meeting Date: {note_date}
Attendees:
Tags:
  - meetings
---
## Topics
-

## Follow Ups
- [ ]
"""


def parse_date(date_str: Optional[str]) -> str:
    """
    Parse a date string in 'YYYY-MM-DD' format or return today's date if None.

    Raises a typer.Exit if the format is invalid.
    """
    if date_str:
        try:
            return datetime.strptime(date_str, '%Y-%m-%d').strftime('%Y-%m-%d')
        except ValueError:
            console.print("[bold red]Invalid date format. Use YYYY-MM-DD.[/bold red]")
            raise typer.Exit(1)
    return datetime.today().strftime('%Y-%m-%d')


def build_contact_map(contacts_folder: pathlib.Path) -> Dict[str, List[str]]:
    """
    Build a mapping from contact filenames to a list of their aliases.

    Aliases are extracted from filenames containing parentheses.
    """
    contact_map: Dict[str, List[str]] = {}
    for file in contacts_folder.rglob('*.md'):
        name = file.stem
        if '(' in name and ')' in name:
            base = name.split('(')[0].strip()
            nickname = name.split('(')[1].split(')')[0].strip()
            contact_map[name] = [base, nickname]
        else:
            contact_map[name] = [name]
    return contact_map


def get_all_contact_names(contact_map: Dict[str, List[str]]) -> List[str]:
    """
    Aggregate all contact names and their aliases into a single list.
    """
    all_names: Set[str] = set()
    for key, aliases in contact_map.items():
        all_names.add(key)
        all_names.update(aliases)
    return list(all_names)


def record_audio(samplerate: int) -> np.ndarray:
    """
    Record audio from the microphone until the user presses ENTER twice:
    once to start and once to stop recording.

    Returns the recorded audio as a numpy array.
    """
    console.print("[bold green]Press ENTER to start recording...[/bold green]")
    input()
    console.print("[bold green]Recording... Press ENTER again to stop.[/bold green]")

    recorded_frames: List[np.ndarray] = []
    start_time = time.time()

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        # Set terminal to raw mode to read single characters without waiting for newline
        tty.setraw(fd)
        with Live("[bold yellow]Recording... 0.0 sec[/bold yellow]", refresh_per_second=4) as live:
            with sd.InputStream(
                    samplerate=samplerate,
                    channels=1,
                    dtype='int16',
                    callback=lambda indata, frames, time_info, status: recorded_frames.append(indata.copy())
            ):
                while True:
                    elapsed = time.time() - start_time
                    live.update(f"[bold yellow]Recording... {elapsed:.1f} sec[/bold yellow]")
                    time.sleep(0.1)
                    # Check if user pressed a key
                    if select.select([sys.stdin], [], [], 0)[0]:
                        c = sys.stdin.read(1)
                        if c in ('\n', '\r'):
                            break
    finally:
        # Restore terminal settings
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    audio_data = np.concatenate(recorded_frames, axis=0)
    return audio_data


def save_audio_to_tempfile(audio_data: np.ndarray, samplerate: int) -> str:
    """
    Save the recorded audio data to a temporary WAV file and return its path.
    """
    audio_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    scipy.io.wavfile.write(audio_path, samplerate, audio_data)
    console.print(f"Audio saved to {audio_path}")
    return audio_path


def transcribe_audio(client: openai.OpenAI, audio_path: str) -> str:
    """
    Transcribe the audio file using OpenAI's Whisper model.
    """
    console.print("[bold green]Transcribing audio...[/bold green]")
    with open(audio_path, "rb") as audio_file:
        transcript_response = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
    return transcript_response.text


def find_matched_contacts(
        transcript: str, all_contact_names: List[str], contact_map: Dict[str, List[str]]
) -> Set[str]:
    """
    Identify contact filenames that best match words in the transcript using fuzzy matching.
    """
    words = transcript.split()
    matched_contacts: Set[str] = set()
    for word in words:
        match = process.extractOne(word, all_contact_names, scorer=fuzz.WRatio)
        if match and match[1] > 80:
            for filename, aliases in contact_map.items():
                if match[0] == filename or match[0] in aliases:
                    matched_contacts.add(filename)
    return matched_contacts


def determine_meeting_topic_and_type(
        transcript: str, matched_contacts: Set[str]
) -> Tuple[str, str]:
    """
    Determine the meeting topic and type based on the transcript and matched contacts.
    """
    lower_transcript = transcript.lower()
    topic_match = re.search(r'topic (of this meeting is|is|:)\s*(.+?)(\.|$)', lower_transcript)
    if topic_match:
        meeting_topic = topic_match.group(2).strip().title()
        meeting_type = 'Meeting'
    elif 'catch up' in lower_transcript and matched_contacts:
        meeting_topic = list(matched_contacts)[0]
        meeting_type = 'Catch Up'
    else:
        meeting_topic = 'General'
        meeting_type = 'Meeting'
    return meeting_topic, meeting_type


def prepare_prompt(
        current_note: str,
        transcript: str,
        template_content: str,
        note_date: str,
        contact_map: Dict[str, List[str]]
) -> str:
    """
    Prepare a prompt string for the language model to generate or refine the markdown note.
    """
    prompt = (
        f"You are an assistant helping create Obsidian markdown notes.\n"
        f"Here is the current note draft:\n{current_note}\n"
        f"Here is the new user input to refine or extend it:\n{transcript}\n"
        f"Here is the template:\n{template_content}\n"
        f"Known contacts and their aliases:\n{contact_map}\n"
        "When generating follow-up tasks, use the Obsidian task format '- [ ] Task description'.\n"
        "If a due date is mentioned in the input, include it in the task using '📅 YYYY-MM-DD'.\n"
        "If an assignee is mentioned, include their wiki-link in the task, e.g. '- [ ] Follow up with [[Alice Smith]] 📅 2025-07-20'.\n"
        "This context is for your guidance and should not appear in the final note.\n"
        "Generate the updated markdown note, integrating the attendees with wiki-links and the new input."
    )
    return prompt


def save_note_file(
        vault_path: pathlib.Path,
        output_folder: str,
        note_date: str,
        meeting_topic: str,
        meeting_type: str,
        current_note: str
) -> pathlib.Path:
    """
    Save the final note markdown file in the specified folder, avoiding filename collisions.
    """
    if meeting_type == 'Catch Up':
        base_filename = f"{note_date} {meeting_topic} Catch Up.md"
    else:
        base_filename = f"{note_date} {meeting_topic} Meeting.md"

    output_path = vault_path / output_folder / base_filename
    counter = 1
    while output_path.exists():
        name_no_ext = base_filename.rsplit('.md', 1)[0]
        output_path = vault_path / output_folder / f"{name_no_ext} ({counter}).md"
        counter += 1

    console.print(f"[bold green]Saving final note as {output_path.name}...[/bold green]")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(current_note, encoding="utf-8")
    return output_path


@app.command()
def create_note(
        contacts_folder: pathlib.Path = typer.Option(..., help="Path to the contacts root folder"),
        vault_path: pathlib.Path = typer.Option(pathlib.Path.home(), help="Path to your Obsidian vault"),
        date: Optional[str] = typer.Option(None, help="Override date in YYYY-MM-DD format"),
        model: str = "gpt-4o",
        output_folder: str = "Meetings",
        samplerate: int = 16000
) -> None:
    """
    Main command to create or refine a meeting note by recording audio, transcribing,
    and generating markdown notes with the help of OpenAI's language models.
    """
    client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    note_date = parse_date(date)

    template_content = TEMPLATE_CONTENT.format(note_date=note_date)

    contact_map = build_contact_map(contacts_folder)
    all_contact_names = get_all_contact_names(contact_map)

    current_note = ""
    while True:
        audio_data = record_audio(samplerate)
        audio_path = save_audio_to_tempfile(audio_data, samplerate)

        transcript = transcribe_audio(client, audio_path)
        console.print("[bold cyan]Transcript:[/bold cyan]", transcript)

        matched_contacts = find_matched_contacts(transcript, all_contact_names, contact_map)
        meeting_topic, meeting_type = determine_meeting_topic_and_type(transcript, matched_contacts)

        prompt = prepare_prompt(current_note, transcript, template_content, note_date, contact_map)

        console.print("[bold green]Generating (or refining) markdown note...[/bold green]")
        chat_response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        current_note = chat_response.choices[0].message.content

        console.print("[bold magenta]\n===== Current Note Draft =====\n[/bold magenta]")
        console.print(Markdown(current_note))

        if not Confirm.ask("Would you like to record another refinement?"):
            break

    output_path = save_note_file(vault_path, output_folder, note_date, meeting_topic, meeting_type, current_note)

    console.print(f"[bold green]Note saved to:[/bold green] {output_path}")


if __name__ == "__main__":
    app()
