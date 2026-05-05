to run the full pipeline please type python -m Src.main in the project's main directory

## API Usage

### Base URL
https://8000-01kpg6x4mnbj6w4w1fx48s4d5t.cloudspaces.litng.ai/predict

### Endpoints

#### `POST /predict`
Returns a survival prediction for a Titanic passenger.

**Request Body Example:**
```json
{
    "Pclass": 3,
    "Sex": 0,
    "Age": 22,
    "SibSp": 1,
    "Parch": 0,
    "Fare": 7.25,
    "Embarked": 2,
    "Ticket": "A/5 21171",
    "Cabin": ""
}
```

**Response:**
```json
{
    "prediction": [0]
}
```
`0` = Did not survive, `1` = Survived

#### `GET /health`
Check if the API and model are running.

**Response:**
```json
{
    "status": "ok",
    "model_loaded": true
}
```

