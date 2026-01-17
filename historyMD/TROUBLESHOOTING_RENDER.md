# 🔧 Troubleshooting Render Deployment

## 🚨 Problem: "Network Error" en Frontend

**Síntoma:** El frontend carga pero al intentar clasificar un email aparece:
```
Analysis Failed
Network Error
```

---

## ✅ Diagnóstico Paso a Paso

### **PASO 1: Verificar que el Backend esté desplegado**

```bash
# Intenta acceder al health endpoint del backend
curl https://spam-detector-api.onrender.com/health
```

**Si obtienes:**
- ✅ `{"status":"healthy"}` → Backend OK, ve al PASO 2
- ❌ `404 Not Found` → Backend NO existe o está mal configurado
- ❌ `503 Service Unavailable` → Backend está "dormido" (Free tier)

---

### **PASO 2: Verificar URLs Correctas**

#### A. Backend URL
1. Ve a tu **Render Dashboard**: https://dashboard.render.com
2. Click en tu servicio de **backend** (Web Service)
3. Copia la URL exacta (arriba derecha), debería ser algo como:
   ```
   https://spam-detector-api-XXXXX.onrender.com
   ```
   **Importante:** Puede incluir un sufijo aleatorio (`-XXXXX`)

#### B. Frontend URL  
1. Click en tu servicio de **frontend** (Static Site)
2. Copia la URL exacta:
   ```
   https://spam-detector-frontend-x4jj.onrender.com
   ```

---

### **PASO 3: Actualizar Variable de Entorno del Frontend**

El frontend necesita saber dónde está el backend:

1. **En Render Dashboard → Frontend Service:**
   - Ve a **"Environment"** (pestaña izquierda)
   
2. **Busca la variable `VITE_API_URL`:**
   - Si NO existe, créala
   - Si existe, verifica que sea correcta

3. **Asegúrate que tenga la URL EXACTA del backend:**
   ```
   VITE_API_URL=https://spam-detector-api-XXXXX.onrender.com
   ```
   ⚠️ **NO incluyas `/health` ni nada más**
   ⚠️ **Usa `https://` (con S)**
   ⚠️ **Sin barra final `/`**

4. **Click "Save Changes"**
   - El frontend se reconstruirá automáticamente (~2 min)

---

### **PASO 4: Actualizar CORS del Backend**

El backend necesita permitir requests desde el frontend:

1. **En Render Dashboard → Backend Service:**
   - Ve a **"Environment"**

2. **Agrega o actualiza `API_CORS_ORIGINS`:**
   ```
   API_CORS_ORIGINS=https://spam-detector-frontend-x4jj.onrender.com
   ```
   
   **Si tienes múltiples URLs, separa con comas:**
   ```
   API_CORS_ORIGINS=https://spam-detector-frontend-x4jj.onrender.com,http://localhost:5173
   ```

3. **Click "Save Changes"**
   - El backend se redespliegará (~2-3 min)

---

### **PASO 5: Verificar que Backend NO esté dormido**

El plan FREE de Render "duerme" el backend después de 15 min de inactividad.

**Síntoma:** Primera request tarda 30-60 segundos

**Solución temporal:**
```bash
# "Despierta" el backend manualmente
curl https://TU-BACKEND-URL.onrender.com/health

# Espera 30-60 segundos si está dormido
# Deberías ver: {"status":"healthy"}
```

**Soluciones permanentes:**
1. **Upgrade a Starter Plan** ($7/mes) → Backend siempre activo
2. **Usar cron externo** (ej: cron-job.org) para hacer ping cada 10 min
3. **Aceptar el delay** en la primera request

---

## 🐛 Errores Comunes

### Error 1: "CORS policy: No 'Access-Control-Allow-Origin'"

**Causa:** Backend no tiene configurado CORS para tu frontend URL

**Solución:**
```bash
# En Backend Environment Variables:
API_CORS_ORIGINS=https://tu-frontend-exacto.onrender.com
```

---

### Error 2: Backend devuelve 404 en `/health`

**Causa:** El servicio de backend no existe o no está corriendo

**Verificar:**
1. En Render Dashboard, ¿ves un servicio tipo "Web Service" con Docker?
2. ¿Está en estado "Live" (verde)?
3. ¿Los logs muestran errores?

**Solución:** Redesplegar backend desde cero (ver DEPLOYMENT_RENDER.md STEP 2)

---

### Error 3: Frontend muestra `undefined` en lugar de URL

**Causa:** Variable `VITE_API_URL` no está configurada o mal escrita

**Verificar:**
```bash
# En Frontend Environment Variables debe existir:
VITE_API_URL=https://tu-backend.onrender.com
```

**Importante:** 
- En Vite, las variables DEBEN empezar con `VITE_`
- Se leen en **BUILD time**, no runtime
- Si cambias la variable, debes **rebuild** el frontend

---

### Error 4: Backend tarda mucho (30+ segundos)

**Causa:** Plan FREE → Backend se duerme después de 15 min sin uso

**Opciones:**
1. **Aceptar delay:** Primera request será lenta
2. **Keep-alive externo:** 
   - Usa https://cron-job.org (gratis)
   - Crea job que haga GET a `/health` cada 10 min
3. **Upgrade:** Starter plan ($7/mes) mantiene backend activo

---

### Error 5: "Failed to load models" en logs del backend

**Causa:** Modelos no se subieron con Git LFS

**Solución:**
```bash
# Localmente:
git lfs track "*.joblib"
git add .gitattributes
git add src/backend/models/*.joblib
git commit -m "Track models with Git LFS"
git push

# Render auto-redespliegará
```

---

## ✅ Checklist Final

Antes de dar por resuelto, verifica:

- [ ] Backend health check responde: 
  ```bash
  curl https://TU-BACKEND.onrender.com/health
  # Debe devolver: {"status":"healthy"}
  ```

- [ ] Frontend tiene `VITE_API_URL` correcta:
  ```
  VITE_API_URL=https://TU-BACKEND.onrender.com
  ```

- [ ] Backend tiene CORS configurado:
  ```
  API_CORS_ORIGINS=https://TU-FRONTEND.onrender.com
  ```

- [ ] Ambos servicios están en estado "Live" (verde) en Render Dashboard

- [ ] No hay errores en los logs de ninguno de los dos servicios

- [ ] Browser console (F12) no muestra errores CORS

---

## 🧪 Test End-to-End

Una vez todo configurado, prueba:

```bash
# 1. Backend directo (sin frontend)
curl -X POST https://TU-BACKEND.onrender.com/api/v1/classify \
  -H "Content-Type: application/json" \
  -d '{"email_text":"URGENT! You won $1,000,000"}'

# Deberías ver JSON con clasificación SPAM/PHISHING
```

```javascript
// 2. Desde Browser Console (F12) en tu frontend
fetch('https://TU-BACKEND.onrender.com/api/v1/classify', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({email_text: 'Test email'})
})
.then(r => r.json())
.then(console.log)

// Si ves la respuesta JSON → TODO OK
// Si ves error CORS → Revisa PASO 4
```

---

## 📞 ¿Aún no funciona?

### Opción 1: Revisar Logs

**Backend:**
1. Render Dashboard → Backend Service → **Logs**
2. Busca errores (palabras clave: `error`, `failed`, `exception`)

**Frontend:**
1. Browser → F12 → **Console**
2. Busca errores de red o CORS

### Opción 2: Redesplegar desde Cero

Si todo falla, borra y recrea:

1. **Borra ambos servicios** en Render Dashboard
2. **Espera 5 minutos** (para que Render limpie)
3. **Sigue DEPLOYMENT_RENDER.md** desde STEP 2

### Opción 3: Deployment Local con Docker

Si Render da problemas, despliega localmente:

```bash
# En proyecto raíz:
docker-compose up --build

# Backend: http://localhost:8000
# Frontend: http://localhost:5173
```

---

## 🎯 Resumen Rápido

```bash
# ✅ URLs que DEBES tener configuradas:

# 1. En Frontend Environment:
VITE_API_URL=https://spam-detector-api-XXXXX.onrender.com

# 2. En Backend Environment:
API_CORS_ORIGINS=https://spam-detector-frontend-x4jj.onrender.com
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=1

# 3. Verificación:
curl https://TU-BACKEND.onrender.com/health
# → {"status":"healthy"}

# 4. Test clasificación:
curl -X POST https://TU-BACKEND.onrender.com/api/v1/classify \
  -H "Content-Type: application/json" \
  -d '{"email_text":"Test"}'
# → JSON con resultados
```

---

**¿Resuelto?** ✅ Marca como cerrado y disfruta tu app en producción!

**¿Aún con problemas?** 🔍 Comparte los logs y te ayudamos.
