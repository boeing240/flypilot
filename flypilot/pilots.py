"""Pilot identities for the league stream: every fly gets a plausible human
name, a nationality code, a racing number and a livery colour.

Identity is a pure function of the fly's seed (plus what's already taken), so a
given fly always keeps the same name. Surnames are deliberately ordinary --
no well-known racing drivers' names.
"""
from __future__ import annotations

import random

# nation -> (first names, surnames)
NATIONS: dict[str, tuple[list[str], list[str]]] = {
    "ESP": (["Mateo", "Javier", "Diego", "Rafael", "Alejandro", "Pablo", "Sergio", "Hugo", "Nicolas", "Ismael"],
            ["Alvarez", "Navarro", "Cordero", "Herrera", "Salinas", "Ibarra", "Beltran", "Quintero", "Ramos", "Velasco"]),
    "ITA": (["Luca", "Matteo", "Giorgio", "Marco", "Davide", "Stefano", "Paolo", "Fabio", "Tommaso", "Riccardo"],
            ["Moretti", "Ferraro", "Bellini", "Conti", "Rinaldi", "Gallo", "Santoro", "Caruso", "Marchetti", "Lombardi"]),
    "GER": (["Lukas", "Jonas", "Felix", "Tobias", "Matthias", "Florian", "Stefan", "Dieter", "Jannik", "Konrad"],
            ["Brandt", "Keller", "Hoffmann", "Vogel", "Lindner", "Krause", "Neumann", "Albrecht", "Reinhardt", "Sommer"]),
    "FRA": (["Julien", "Baptiste", "Antoine", "Mathieu", "Romain", "Thibault", "Maxime", "Quentin", "Remi", "Anselme"],
            ["Laurent", "Moreau", "Fabre", "Girard", "Lefevre", "Roux", "Perrin", "Delacroix", "Marchand", "Garnier"]),
    "GBR": (["Oliver", "Harry", "Callum", "Jack", "Rhys", "Liam", "Ewan", "Connor", "Archie", "Percy"],
            ["Whitmore", "Hargreaves", "Pemberton", "Ashworth", "Fletcher", "Thornton", "Blackwood", "Langley", "Sutcliffe", "Holloway"]),
    "JPN": (["Kenji", "Haruto", "Ren", "Takumi", "Yuto", "Daichi", "Shota", "Ryota", "Sora", "Hiroki"],
            ["Tanaka", "Watanabe", "Nakamura", "Fujimoto", "Hayashi", "Ishikawa", "Matsuda", "Ogawa", "Morita", "Yamashita"]),
    "BRA": (["Thiago", "Gabriel", "Rodrigo", "Caio", "Bruno", "Leandro", "Vinicius", "Matheus", "Edson", "Lucio"],
            ["Souza", "Barbosa", "Teixeira", "Carvalho", "Nogueira", "Moreira", "Cardoso", "Pacheco", "Duarte", "Bastos"]),
    "SWE": (["Erik", "Oskar", "Anders", "Henrik", "Emil", "Viktor", "Axel", "Nils", "Gunnar", "Stellan"],
            ["Lindqvist", "Bergstrom", "Holmgren", "Sandberg", "Ekstrom", "Nyberg", "Dahlgren", "Forsberg", "Axelsson", "Wikander"]),
    "FIN": (["Eero", "Mikko", "Juha", "Teemu", "Ville", "Antti", "Jari", "Oskari", "Topi", "Lauri"],
            ["Virtanen", "Korhonen", "Lehtinen", "Laine", "Heikkinen", "Niemi", "Koskinen", "Hakala", "Mattila", "Turunen"]),
    "NED": (["Bram", "Sander", "Joris", "Daan", "Ruben", "Thijs", "Wouter", "Gijs", "Pim", "Joost"],
            ["van Dijk", "Bakker", "Visser", "Smit", "Mulder", "Kuiper", "Hoekstra", "Brouwer", "Dekker", "Vermeulen"]),
    "USA": (["Cole", "Wyatt", "Brody", "Garrett", "Dustin", "Colton", "Tanner", "Mason", "Dale", "Boone"],
            ["Callahan", "Prescott", "Whitaker", "Sullivan", "Rourke", "Hollister", "Danforth", "McAllister", "Branigan", "Kowalski"]),
    "AUS": (["Jarrod", "Lachlan", "Hamish", "Brodie", "Mitch", "Declan", "Angus", "Flynn", "Jett", "Rory"],
            ["Kingsley", "Treloar", "Mackenzie", "Gallagher", "Bannister", "Aldridge", "Cosgrove", "Hartigan", "Penrose", "Tregear"]),
    "MEX": (["Emiliano", "Santiago", "Fernando", "Ricardo", "Joaquin", "Mauricio", "Octavio", "Ignacio", "Leonel", "Ulises"],
            ["Guerrero", "Mendoza", "Palacios", "Villarreal", "Cisneros", "Duran", "Olvera", "Segura", "Zavala", "Contreras"]),
    "POL": (["Piotr", "Marek", "Tomasz", "Kuba", "Wojtek", "Lech", "Bartek", "Radek", "Maciej", "Jacek"],
            ["Nowak", "Kaminski", "Zielinski", "Wozniak", "Kozlowski", "Jablonski", "Mazur", "Sikora", "Baran", "Czerwinski"]),
    "IND": (["Arjun", "Rohan", "Vikram", "Karan", "Aditya", "Rahul", "Nikhil", "Sameer", "Dev", "Kabir"],
            ["Mehta", "Kapoor", "Iyer", "Banerjee", "Chauhan", "Desai", "Naidu", "Bhatt", "Sethi", "Rastogi"]),
    "TUR": (["Emre", "Burak", "Kerem", "Mert", "Cem", "Onur", "Baris", "Tolga", "Selim", "Kaan"],
            ["Yilmaz", "Demir", "Aksoy", "Korkmaz", "Arslan", "Tuna", "Erdem", "Sahin", "Ozkan", "Kaplan"]),
}

# Livery colours: readable on the dark stream background and distinct from each
# other. Gold is reserved for the Legend, so nothing here sits near it.
PALETTE = [
    "#ff4d4d", "#ff8c1a", "#a6e22e", "#2ecc71", "#1abc9c", "#29b6f6", "#4f7cff", "#8e6bff",
    "#d35cff", "#ff5cc8", "#ff9a9a", "#b8c2d6", "#e0a96d", "#7ee8fa", "#c0ff7a",
]
LEGEND_COLOR = "#ffd700"


def make_identity(seed: int, taken_names: set[str], taken_numbers: set[int],
                  taken_colors: set[str], legend: bool = False) -> dict:
    rng = random.Random(seed)
    for _ in range(200):
        nation = rng.choice(sorted(NATIONS))
        firsts, lasts = NATIONS[nation]
        name = f"{rng.choice(firsts)} {rng.choice(lasts)}"
        if name not in taken_names:
            break
    else:
        name = f"{name} {rng.randrange(2, 99)}"

    if legend:
        number = 1
        color = LEGEND_COLOR
    else:
        free_numbers = [n for n in range(2, 100) if n not in taken_numbers]
        number = rng.choice(free_numbers)
        free_colors = [c for c in PALETTE if c not in taken_colors] or PALETTE
        color = rng.choice(free_colors)
    return {"name": name, "nation": nation, "number": number, "color": color, "legend": legend}
