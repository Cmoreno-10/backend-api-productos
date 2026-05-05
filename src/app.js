import express from 'express';
import cors from 'cors';
import path from 'path';
import { fileURLToPath } from 'url';
import productosRoutes from './routes/productos.routes.js';
import productosDbRoutes from './routes/productos.db.routes.js';

const app = express();

// Middlewares
app.use(cors());
app.use(express.json());

// Configuración de archivos estáticos
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

app.use(express.static(path.resolve(__dirname, '../public')));

// Rutas
app.use('/api/productos', productosRoutes);
app.use('/api/productos-db', productosDbRoutes);

export default app;
