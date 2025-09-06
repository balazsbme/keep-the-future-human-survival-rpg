import os
from typing import List

from rpg.character import MarkdownCharacter


def load_characters() -> List[MarkdownCharacter]:
    base = os.path.join(os.path.dirname(__file__), "characters")
    return [
        MarkdownCharacter("Alice", os.path.join(base, "alice.md")),
        MarkdownCharacter("Bob", os.path.join(base, "bob.md")),
    ]


def main() -> None:
    characters = load_characters()
    for idx, char in enumerate(characters, 1):
        print(f"{idx}. {char.name}")
    choice = int(input("Choose a character: ")) - 1
    char = characters[choice]
    options = char.generate_questions()
    for idx, q in enumerate(options, 1):
        print(f"{idx}. {q}")
    question = options[int(input("Choose a question: ")) - 1]
    print(char.answer_question(question))


if __name__ == "__main__":
    main()
