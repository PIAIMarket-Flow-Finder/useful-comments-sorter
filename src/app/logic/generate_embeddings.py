import json
from sentence_transformers import SentenceTransformer



def generate_embeddings(comments):

    # Charger le modèle
    model = SentenceTransformer("sentence-transformers/LaBSE", device="cuda")

    # Extraire la liste de commentaires
    comments = comments.get("comments", [])

    if not isinstance(comments, list):
            raise ValueError("Expected 'comments' to be a list.")

    # Déterminer si c'est une liste de chaînes ou d'objets
    if all(isinstance(el, str) for el in comments):
        texts = comments
        enriched_comments = [{"content": text} for text in texts]
    elif all(isinstance(el, dict) and "content" in el for el in comments):
        texts = [el["content"] for el in comments]
        enriched_comments = comments
    else:
        raise ValueError("Each comment must be either a string or a dict with a 'content' key.")

    # Générer les vecteurs
    vectors = model.encode(texts, show_progress_bar=True, batch_size=32, convert_to_tensor=True)

    for el, vec in zip(enriched_comments, vectors):
        el["vector"] = vec.tolist()

    return {"comments": enriched_comments}