import { Router } from 'express';

const router = Router();

router.get('/', (req, res) => {
  res.json([
    { nombre: 'Revitalift', foto: 'revitalift.avif', votacion: 120 },
    { nombre: 'Elseve', foto: 'elseve.jpg', votacion: 98 },
    { nombre: 'Infallible', foto: 'infallible.jfif', votacion: 150 },
    { nombre: 'Elvive', foto: 'elvive.avif', votacion: 87 },
    { nombre: 'True Match', foto: 'truematch.webp', votacion: 110 }
  ]);
});

export default router;
