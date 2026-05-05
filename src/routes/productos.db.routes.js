import { Router } from 'express';
import { pool } from '../config/db.js';

const router = Router();

router.get('/', async (req, res) => {
  try {
    const [rows] = await pool.query(
      'SELECT nombre, foto, votacion FROM productos'
    );

    res.json(rows);
  } catch (error) {
    console.error('ERROR MYSQL:', error);

    res.status(500).json({
      error: 'Error al consultar MySQL',
      detalle: error.message,
      codigo: error.code
    });
  }
});

export default router;
