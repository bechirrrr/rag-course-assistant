"""Tests de la fonction nettoyer() d'ingest.py."""
from ingest import nettoyer


def test_supprime_les_pieds_de_slide():
    texte = "Introduction\nIDS 4/IGL 5\nDr. Asma MEKKI\nDu déclaratif au procédural:"
    assert nettoyer(texte) == "Introduction"


def test_supprime_dates_et_numeros_seuls():
    texte = "31/10/2026\nLa tokenisation découpe un texte.\n183"
    assert nettoyer(texte) == "La tokenisation découpe un texte."


def test_garde_les_phrases_qui_contiennent_des_nombres():
    texte = "Les données structurées représentent 20% des données."
    assert nettoyer(texte) == texte


def test_supprime_lignes_vides_et_espaces():
    assert nettoyer("  \n\n  Lemmatisation  \n\n") == "Lemmatisation"


def test_page_vide():
    assert nettoyer("IDS 4/IGL 5\n12\n") == ""
