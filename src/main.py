import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query

from src.models import (
    HealthResponse,
    ItemCreate,
    MessageResponse,
    RatingCreate,
    RecommendationItem,
    UserCreate,
)
from src.recommender import RecommenderSystem


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
REVIEWS_PATH = os.path.join(DATA_DIR, "amazon_reviews_automotive_sample.csv")
META_PATH = os.path.join(DATA_DIR, "amazon_meta_automotive_sample.csv")

recommender = RecommenderSystem()
users: dict[str, UserCreate] = {}
items: dict[str, ItemCreate] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if recommender.total_reviews == 0:
        try:
            recommender.load_data(REVIEWS_PATH, META_PATH)
        except FileNotFoundError:
            pass
    yield


DESCRIPTION = """

Bem-vindo à documentação oficial do **Sistema de Recomendação Automotiva**! 

Esta API utiliza técnicas de Inteligência Artificial baseadas em **Filtragem Colaborativa (Similaridade de Cosseno)** sobre o dataset Amazon Automotive para gerar recomendações personalizadas de peças e acessórios automotivos.

* Nome: Raul Falluh Fragoso de Mendonça
* Matrícula: 22300926
* Instituição: Centro Universitário de Brasília
* Curso: Ciência da Computação
* Período: 7º Semestre
* Matéria: Desenvolvimento de sistemas de IA
* Professor: Fábio Oliveira Guimarães 

## Principais Funcionalidades

* **Gestão de Usuários**: Cadastre e liste usuários no sistema.
* **Gestão de Itens**: Adicione novas peças automotivas ao catálogo.
* **Avaliações**: Registre as avaliações dos usuários para os itens (alimentando a IA).
* **Recomendações (IA)**: Obtenha recomendações personalizadas com base no histórico de avaliações do usuário ou os itens mais populares (Cold Start).

## Como Testar

Você pode testar a API diretamente aqui no Swagger! Siga o fluxo básico:
1. **Cadastre um usuário** em `POST /users/`.
2. **Cadastre um item** em `POST /items/`.
3. **Avalie um item** em `POST /ratings/` (use o usuário e item cadastrados).
4. **Obtenha recomendações** em `GET /recommendations/{user_id}`.
"""

app = FastAPI(
    title="Sistema de Recomendação Automotiva API",
    description=DESCRIPTION,
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", tags=["Util"])
def read_root():
    """Retorna uma mensagem de boas-vindas."""
    return {
        "message": "Bem-vindo à documentação oficial do Sistema de Recomendação Automotiva API",
        "version": "1.0.0",
    }


@app.get("/health", response_model=HealthResponse, tags=["Util"])
def health_check():
    """Verifica o status do sistema de recomendação e retorna estatísticas."""
    return recommender.stats()


@app.post("/users", response_model=UserCreate, tags=["Usuarios"])
def create_user(user: UserCreate):
    """Cadastra um novo usuário no sistema."""
    if user.user_id in users:
        raise HTTPException(status_code=400, detail="Usuario ja cadastrado")
    users[user.user_id] = user
    return user


@app.post("/items", response_model=ItemCreate, tags=["Itens"])
def create_item(item: ItemCreate):
    """Cadastra uma nova peça automotiva no sistema."""
    if item.parent_asin in items:
        raise HTTPException(status_code=400, detail="Item ja cadastrado")
    items[item.parent_asin] = item
    return item


@app.get("/users", response_model=list[UserCreate], tags=["Usuarios"])
def get_users():
    """Retorna a lista de usuários cadastrados manualmente nesta sessão."""
    return list(users.values())


@app.get("/items", response_model=list[ItemCreate], tags=["Itens"])
def get_items():
    """Retorna a lista de itens cadastrados manualmente nesta sessão."""
    return list(items.values())


@app.post("/ratings", response_model=MessageResponse, tags=["Avaliacoes"])
def add_rating(rating: RatingCreate):
    """Registra a avaliação de um usuário para um item e atualiza a Inteligência Artificial."""
    recommender.add_rating(rating.user_id, rating.parent_asin, rating.rating)
    return MessageResponse(
        message=(
            f"Nota {rating.rating} do usuario {rating.user_id} "
            f"para o item {rating.parent_asin} registrada."
        )
    )


@app.get(
    "/recommendations/{user_id}",
    response_model=list[RecommendationItem],
    tags=["Recomendacoes"],
)
def get_recommendations(
    user_id: str,
    n: int = Query(5, ge=1, le=50, description="Quantidade de itens a recomendar"),
):
    return [RecommendationItem(**r) for r in recommender.recommend(user_id, n)]
