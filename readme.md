# E-Commerce Recommendation System

A full-stack application featuring a FastAPI backend and Next.js frontend. The backend provides personalized product recommendations using semantic similarity, machine learning, and LLM-powered explanations. The frontend offers a modern dashboard for catalog management, user behavior tracking, and AI-driven recommendations.

## 🚀 Demo

### 📺 Video Walkthrough

Watch a full demo of the application:
[Watch Demo](https://www.youtube.com/watch?v=<your-youtube-video-id>)

## 📊 Recommendation Algorithm

![Recommendation Flow](https://drive.google.com/uc?export=view&id=1Bmpvw6egiruBqJfe_xxl6pitYFvZFz3L)

### How It Works

1. **User Behavior Retrieval**: Fetches last 10 products user interacted with
2. **Category Focus**: Identifies category of most recently viewed product
3. **Multi-Stage Filtering**:
   - Price range: ±12.5% of category average
   - Similarity: Top 70% (≥30th percentile)
   - Rating: Above median rating
4. **Scoring**: Combined score calculation
   - 50% Semantic Similarity
   - 30% Rating (normalized)
   - 20% Price Value (normalized)
5. **Explanation Generation**: LLM creates human-readable reasoning
6. **Storage**: Saves recommendations to MongoDB

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- MongoDB
- Groq API Key
- Node.js 18.17 or later ([Download](https://nodejs.org/))
- npm 9.0 or later (comes with Node.js)
- Git ([Download](https://git-scm.com/))

### 🚀 Backend Setup

1. Clone the repository and enter the backend folder:
    ```bash
    git clone <repository-url>
    cd backend
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Create `.env` file:
    ```env
    MONGO_CONNECTION_STRING=mongodb://localhost:27017
    GROQ_API_KEY=your_groq_api_key_here
    ```

4. Run the server:
    ```bash
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
    ```

---

### 🌐 Frontend Setup

1. Enter the frontend folder:
    ```bash
    cd ../frontend
    ```

2. Install dependencies:
    ```bash
    npm install
    ```

3. Create a `.env.local` file in the frontend root:
    ```bash
    cp .env.example .env.local
    ```

4. Edit `.env.local` and set your API URL:
    ```env
    NEXT_PUBLIC_API_URL=http://localhost:8000
    ```

5. Run the development server:
    ```bash
    npm run dev
    ```

    Open [http://localhost:3000](http://localhost:3000) in your browser.

6. Build for production:
    ```bash
    npm run build
    npm start
    ```

## 🛠️ API Endpoints

The frontend integrates with a FastAPI backend. All 14 API endpoints are fully integrated:

### Recommendations APIs
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/upload/products` | Upload products CSV |
| POST | `/api/upload/user-behavior` | Upload user behavior CSV |
| GET | `/api/recommendations/{user_id}` | Generate recommendations |
| GET | `/api/recommendations/stored/{user_id}` | Get stored recommendations |

### Products APIs
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/products/add_product` | Add new product |
| GET | `/products/get_products` | Get all products |
| GET | `/products/get_product_id/{product_id}` | Get product by ID |
| PUT | `/products/update_product/{product_id}` | Update product |
| DELETE | `/products/delete_product/{product_id}` | Delete product |

### User Behavior APIs
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/behavior/add_behavior` | Add behavior record |
| GET | `/behavior/get_behaviors` | Get all behaviors |
| GET | `/behavior/get_behavior/{behavior_id}` | Get behavior by ID |
| PUT | `/behavior/update_behavior/{behavior_id}` | Update behavior |
| DELETE | `/behavior/delete_behavior/{behavior_id}` | Delete behavior |

**📚 Full API Documentation:** See [API_INTEGRATION.md](./API_INTEGRATION.md)

## 🛠️ Tech Stack

### Backend
- **Framework**: FastAPI 0.115.9
- **Database**: MongoDB (PyMongo 4.11.2)
- **ML/AI**:
  - PyTorch 2.7.0
  - Transformers 4.36.0 (Sentence Transformers)
  - Scikit-learn 1.6.1
- **LLM**: LangChain + Groq (LLaMA 3.3 70B)

### Frontend
- **Framework**: [Next.js 16.0.7](https://nextjs.org/) (App Router)
- **UI Library**: [React 19.2.0](https://react.dev/)
- **Language**: [TypeScript 5](https://www.typescriptlang.org/)
- **Styling**: [Tailwind CSS 4.1.9](https://tailwindcss.com/)
- **UI Components**: [Radix UI](https://www.radix-ui.com/)
- **Icons**: [Lucide React](https://lucide.dev/)
- **Notifications**: [Sonner](https://sonner.emilkowal.ski/)
- **Forms**: [React Hook Form](https://react-hook-form.com/) + [Zod](https://zod.dev/)
- **API Client**: Native Fetch with TypeScript
- **Data Format**: JSON + CSV upload support
- **Error Handling**: Comprehensive error boundaries
- **State Management**: React Hooks
- **Build Tool**: [Turbopack](https://turbo.build/)
- **Linting**: [ESLint](https://eslint.org/)
- **Package Manager**: npm
- **Version Control**: Git

## 📁 Project Structure

### Backend (`e-Commerce-Recommender-backend/`)

```
e-Commerce-Recommender-backend/
├── main.py                 # FastAPI application entry point
├── routers/
│   ├── recommend.py        # Recommendation endpoints
│   ├── product.py          # Product CRUD endpoints
│   └── behavior.py         # User behavior endpoints
├── utils/
│   └── recommender.py      # Core recommendation logic
├── requirements.txt        # Python dependencies
├── dockerfile              # Docker configuration
├── .env                    # Environment variables
└── readme.md               # This file
```

### Frontend (`recommendai/`)

```
recommendai/
├── app/                          # Next.js App Router
│   ├── dashboard/                # Dashboard pages
│   │   ├── catalog/              # Product catalog management
│   │   ├── behaviour/            # User behavior tracking
│   │   ├── recommendations/      # AI recommendations
│   │   └── layout.tsx            # Dashboard layout
│   ├── globals.css               # Global styles
│   ├── layout.tsx                # Root layout
│   └── page.tsx                  # Home page
├── components/                   # React components
│   ├── ui/                       # Shadcn UI components
│   ├── app-sidebar.tsx           # Navigation sidebar
│   ├── editable-table.tsx        # Data table with CRUD
│   ├── file-uploader.tsx         # CSV file uploader
│   ├── navbar.tsx                # Navigation bar
│   ├── pagination.tsx            # Pagination component
│   └── upload-dialog.tsx         # Upload dialog
├── lib/                          # Utilities and helpers
│   ├── api.ts                    # API client (14 endpoints)
│   ├── mock-data.ts              # Mock data for development
│   ├── types.ts                  # TypeScript types
│   └── utils.ts                  # Utility functions
├── hooks/                        # Custom React hooks
│   └── use-mobile.tsx            # Mobile detection hook
├── public/                       # Static assets
│   ├── icon.svg                  # App icon
│   └── ...
├── .env.example                  # Environment variables template
├── .env.local                    # Local environment variables
├── package.json                  # Dependencies
├── tsconfig.json                 # TypeScript configuration
├── tailwind.config.ts            # Tailwind CSS configuration
└── next.config.ts                # Next.js configuration
```

## 🔍 Recommendation Details

### Filtering Criteria

- **Price Range**: ±12.5% of category average
- **Similarity Threshold**: 30th percentile (top 70%)
- **Rating Filter**: Above median rating

### Scoring Formula

```
score = 0.5 × similarity + 0.3 × rating_norm + 0.2 × price_norm
```

### Models Used

- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
- **LLM**: `llama-3.3-70b-versatile` (via Groq)

## 📝 Sample Data Format

### Products CSV
```csv
product_id,name,category,price,rating,brand,features,stock
P001,Product Name,Electronics,999.99,4.5,Brand,Features,50
```

### User Behavior CSV
```csv
user_id,product_id,action,timestamp
U1001,P081,view,2024-01-10 14:05:00
U1001,P083,add_to_cart,2024-01-10 14:20:00
```

### Supported Actions:
```
- `view` - User viewed the product
- `add_to_cart` - User added product to cart
- `purchase` - User purchased the product
```