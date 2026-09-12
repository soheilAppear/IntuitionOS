"""Natural-language file requests must not become unrelated shell commands."""

import pytest

from core.command_resolver import CommandResolver, IntuitionCommandProvider, StaticCommandProvider
from core.file_intents import is_file_request, try_file_intent
from core.os_intents import _try_os_intent, is_app_command


@pytest.mark.parametrize("phrase", [
    "write a Python program in hello.py that prints hello",
    "write Python code into hello.py",
    "write a file note.txt with my shopping list",
])
def test_code_writing_is_preserved_for_model_not_shell(phrase):
    assert is_file_request(phrase)
    assert is_app_command(phrase)
    assert _try_os_intent(phrase) is None


@pytest.mark.parametrize("phrase", [
    "Change live_calculator.py so running it prints add(10, 20) instead. Keep add(a, b) unchanged.",
    "edit hello.py to print('hello') instead",
    "modify CreatedFolder/calculator.py so add(a, b) returns a + b",
    "update the file report.md with the new results",
    'Please rewrite "CreatedFolder/My Code.py" using a function that returns (1, 2)',
    "Could you change my script.py by replacing print('old') with print('new')?",
    r"change C:\work\CreatedFolder\hello.py so it prints 'ready'",
])
def test_natural_edits_keep_filename_and_code_punctuation_for_the_model(phrase):
    resolver = CommandResolver([
        IntuitionCommandProvider(),
        StaticCommandProvider(["change", "edit", "modify", "update", "rewrite"], case_sensitive=False),
    ], shell="cmd")
    assert is_file_request(phrase)
    assert is_app_command(phrase)
    assert try_file_intent(phrase) is None
    assert _try_os_intent(phrase) is None
    resolution = resolver.resolve(phrase)
    assert resolution.original == phrase
    assert resolution.status == "exact"
    assert resolution.namespace == "intuitionos"
    assert resolution.candidates == []


@pytest.mark.parametrize("phrase", [
    "change logon /query", "change port /query", "edit notes.txt", "edit --help",
    "update script.py --check", "rewrite input.py output.py",
    "/exec change live_calculator.py so it prints add(10, 20)",
    'git commit -m "change hello.py so it prints hello"',
])
def test_edit_guard_preserves_literal_shell_commands_and_explicit_exec(phrase):
    assert not is_file_request(phrase)
    assert not is_app_command(phrase)


@pytest.mark.parametrize("phrase,name,directory", [
    ("make a new file 1.py in desktop directory", "1.py", "desktop"),
    ("create an empty file 1.py on my Desktop", "1.py", "desktop"),
    ("make file Report.py in the documents folder", "Report.py", "documents"),
    ("create a file named Notes.txt in Downloads", "Notes.txt", "downloads"),
    ("Create a new empty file called 'My Script.py' on the Desktop.", "My Script.py", "desktop"),
    ('Please make a file "My Notes.txt" in my Documents directory', "My Notes.txt", "documents"),
    ("Can you please create a file Main.py in the project directory for me?", "Main.py", "project"),
    ("could you make a new file README in the current directory please", "README", "project"),
    ("create file x.py in current working directory", "x.py", "project"),
    ("make an empty file Main.py", "Main.py", "project"),
    ("make a new file call soheil.txt", "soheil.txt", "project"),
    ("create a file hello.py in CreatedFolder", "hello.py", "project"),
    ("create a file hello.py in the CreatedFolder directory", "hello.py", "project"),
    ('create file "Two  Spaces.py"', "Two  Spaces.py", "project"),
    ("create a file 1.py.", "1.py", "project"),
    ("create a file 1.py?", "1.py", "project"),
    ("create a file 1.py!", "1.py", "project"),
    ('create file "Wow!.txt"', "Wow!.txt", "project"),
])
def test_complete_file_requests_route_without_changing_the_filename(phrase, name, directory):
    expected = "create_empty_file", {"name": name, "directory": directory}
    assert try_file_intent(phrase) == expected
    assert _try_os_intent(phrase) == expected
    assert is_file_request(phrase)
    assert is_app_command(phrase)


@pytest.mark.parametrize("phrase", [
    "make a new file 1.py in desktop directory and show the battery",
    "create file x.py and lock the computer",
    "make a file screenshot.py with code that takes a screenshot",
    "create a file 1.py containing print('hello')",
    "make an empty file x.py in Documents and open it",
    "create a file called 1.py with this content:\nprint('hello')",
    "please make a new Python file on desktop that prints hello",
    "create a new folder on Desktop and read clipboard",
    "make a document about network status",
    "make a new file 1.py in an unsupported directory",
    "create a file",
    'make a new file "unterminated name.py in Desktop',
])
def test_unsupported_file_requests_remain_intact_for_the_ai(phrase):
    assert try_file_intent(phrase) is None
    assert _try_os_intent(phrase) is None
    assert is_file_request(phrase)
    assert is_app_command(phrase)


@pytest.mark.parametrize("phrase", [
    "make build", "make -j", "make -f Makefile", "/exec make a new file 1.py",
    "make file.o", "make directory/file.o", "make file-target", "make files.o",
    "makecab a new file 1.py in desktop directory",
    "makecab source.txt archive.cab", "git commit -m 'make a new file'",
    "echo create a file x.py", "python create_file.py", "create-react-app my-app",
])
def test_literal_commands_are_not_file_requests(phrase):
    assert try_file_intent(phrase) is None
    assert not is_file_request(phrase)
    assert not is_app_command(phrase)
