import re
import fitz  # PyMuPDF

# Lignes répétées sur toutes les pages (pied de slide)
LIGNES_A_SUPPRIMER = {"IDS 4/IGL 5", "Dr. Asma MEKKI"}


def nettoyer(texte):
    lignes_gardees = []
    for ligne in texte.splitlines():
        ligne = ligne.strip()
        if ligne == "":
            continue                                   # ligne vide
        if ligne in LIGNES_A_SUPPRIMER:
            continue                                   # pied de slide
        if re.fullmatch(r"\d{2}/\d{2}/\d{4}", ligne):
            continue                                   # date, ex. 31/10/2026
        if re.fullmatch(r"\d+", ligne):
            continue                                   # numéro seul, ex. 182
        lignes_gardees.append(ligne)
    return "\n".join(lignes_gardees)


doc = fitz.open("data/raw/nlp/Cours_TALN_chap_4.pdf")
page = doc[1]  # la 2e page (Python compte à partir de 0)
texte = page.get_text()

print("========== AVANT ==========")
print(texte)
print("========== APRÈS ==========")
print(nettoyer(texte))
print("===========================")
print("Caractères avant :", len(texte))
print("Caractères après :", len(nettoyer(texte)))
