# 🚨 CREAR BACKEND EN RENDER - AHORA

## ❌ Problema Confirmado

El backend **NO EXISTE** en Render. El header `x-render-routing: no-server` lo confirma.

**Necesitas crear el servicio backend manualmente.**

---

## 🚀 Crear Backend - Paso a Paso (5 minutos)

### **PASO 1: Ir a Render Dashboard**

1. Abre: https://dashboard.render.com
2. Inicia sesión con tu cuenta de GitHub

---

### **PASO 2: Crear Nuevo Web Service**

1. Click en el botón azul **"New +"** (arriba derecha)
2. Selecciona **"Web Service"**

---

### **PASO 3: Conectar Repositorio**

1. En la lista de repositorios, busca: **`ML-Spam-Phising-Detector`**
2. Click en **"Connect"** al lado del repositorio

**Si no aparece el repositorio:**
- Click en "Configure account" → Autoriza Render para acceder a tus repos
- Refresca la página y vuelve a buscar

---

### **PASO 4: Configurar el Servicio**

Llena el formulario con estos valores EXACTOS:

#### **Información Básica:**
```
Name: spam-detector-api
Region: Oregon (US West) - o el más cercano a ti
Branch: main
```

#### **Root Directory:**
```
src/backend
```
⚠️ **IMPORTANTE:** Escribe exactamente `src/backend` (sin `/` inicial)

#### **Environment:**
```
Docker
```
⚠️ **IMPORTANTE:** Selecciona "Docker", NO "Python"

#### **Plan:**
```
Free
```

---

### **PASO 5: Configuración Avanzada (Advanced)**

Click en **"Advanced"** para expandir opciones adicionales:

#### **Docker Configuration:**
```
Dockerfile Path: Dockerfile
Docker Context: .
Docker Command: (dejar vacío)
```

#### **Health Check:**
```
Health Check Path: /health
```

---

### **PASO 6: Variables de Entorno**

Click en **"Add Environment Variable"** y agrega estas 4 variables:

```env
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=1
API_CORS_ORIGINS=https://spam-detector-frontend-x4jj.onrender.com
```

**Cómo agregar cada variable:**
1. Click "Add Environment Variable"
2. Key: `API_HOST` | Value: `0.0.0.0`
3. Click "Add Environment Variable" de nuevo
4. Key: `API_PORT` | Value: `8000`
5. Repite para las otras dos

---

### **PASO 7: Crear el Servicio**

1. Scroll hasta el final del formulario
2. Click en el botón azul grande **"Create Web Service"**

---

### **PASO 8: Esperar el Deployment**

Verás la pantalla de logs en tiempo real:

```
=== Deploying web service ===
Cloning repository...
Building Docker image...
Installing dependencies...
Loading models...
Starting application...
✓ Application startup complete
✓ Health check passed
=== Deploy successful ===
```

**Tiempo estimado: 3-5 minutos**

---

### **PASO 9: Obtener la URL del Backend**

Una vez el deployment termine:

1. En la parte superior de la página verás una URL como:
   ```
   https://spam-detector-api.onrender.com
   ```
   O puede tener un sufijo:
   ```
   https://spam-detector-api-xxxxx.onrender.com
   ```

2. **COPIA ESTA URL EXACTA** (la necesitarás en el siguiente paso)

---

### **PASO 10: Verificar que Funciona**

Abre una terminal y ejecuta:

```bash
# Reemplaza con TU URL exacta del paso anterior
curl https://spam-detector-api.onrender.com/health

# Debe devolver:
{"status":"healthy"}
```

Si devuelve `{"status":"healthy"}` → **✅ BACKEND FUNCIONA**

---

## 🔧 PASO 11: Actualizar Frontend con la URL Correcta

Ahora que el backend existe, actualiza el frontend:

### En Render Dashboard:

1. Click en **"spam-detector-frontend"** en el dashboard
2. Click en **"Environment"** (menú izquierdo)
3. Busca la variable `VITE_API_URL`
4. **Edita el valor** con la URL EXACTA del backend (del PASO 9)
   ```
   https://spam-detector-api.onrender.com
   ```
   O si tu URL tiene sufijo:
   ```
   https://spam-detector-api-xxxxx.onrender.com
   ```
5. Click **"Save Changes"**

El frontend se reconstruirá automáticamente (~2 minutos).

---

## ✅ PASO 12: Verificación Final

Después de 2 minutos:

1. Abre: https://spam-detector-frontend-x4jj.onrender.com
2. En el header, verifica: **API Status = 🟢 Online**
3. Abre consola del navegador (F12)
4. Busca la línea:
   ```
   🔗 API Base URL: https://spam-detector-api.onrender.com
   ```
5. Pega un email de prueba:
   ```
   URGENT! You won $1,000,000! Click here NOW!
   ```
6. Click **"Analyze Email"**
7. Verifica que aparezcan los resultados

**Si todo funciona → ✅ ¡ÉXITO! Tu app está en producción**

---

## 🐛 Troubleshooting

### Error: "Repository not found"
**Solución:** Ve a https://github.com/settings/installations
- Find "Render"
- Click "Configure"
- En "Repository access", selecciona tu repositorio
- Save

### Error: "Docker build failed"
**Causa común:** Git LFS no está instalado o modelos no se descargaron

**Solución:**
```bash
# En tu máquina local:
cd /home/air/Escritorio/AIR/Studies/AI-Path/Ml-Engineer
git lfs pull
git add src/backend/models/
git commit -m "Ensure models are tracked with Git LFS"
git push

# Luego en Render:
Manual Deploy → Deploy latest commit
```

### Error: "Health check failed"
**Verifica logs:**
1. Render Dashboard → Backend Service → Logs
2. Busca líneas con `ERROR` o `FAILED`
3. Si ves "Models not found" → Problema con Git LFS (ver arriba)
4. Si ves "Port already in use" → Reinicia el servicio

### Backend se creó pero devuelve 404 en /health
**Verifica en logs que diga:**
```
Application startup complete
Uvicorn running on http://0.0.0.0:8000
```

Si NO aparece, el problema es el CMD del Dockerfile (ya está arreglado en el último commit).

---

## 📊 Resumen Visual

```
┌─────────────────────────────────────────┐
│   ANTES (❌ No funciona)                │
├─────────────────────────────────────────┤
│ Frontend: ✅ Online                      │
│ Backend:  ❌ NO EXISTE                   │
│ Resultado: Network Error                │
└─────────────────────────────────────────┘

            ⬇️ DESPUÉS DE ESTOS PASOS ⬇️

┌─────────────────────────────────────────┐
│   DESPUÉS (✅ Funciona)                 │
├─────────────────────────────────────────┤
│ Frontend: ✅ Online                      │
│ Backend:  ✅ Online                      │
│ Conexión: ✅ Funcionando                │
│ Resultado: Clasificación exitosa        │
└─────────────────────────────────────────┘
```

---

## ⏱️ Timeline Total

```
T+0min:  Crear servicio en Render
T+1min:  Render clona repo y detecta Dockerfile
T+3min:  Docker build completo
T+5min:  Backend LIVE ✅
T+7min:  Actualizar VITE_API_URL en frontend
T+9min:  Frontend rebuild completo
T+10min: TODO FUNCIONA ✅🎉
```

---

## 🎯 Checklist Rápido

- [ ] Ir a dashboard.render.com
- [ ] New + → Web Service
- [ ] Conectar repo ML-Spam-Phising-Detector
- [ ] Name: spam-detector-api
- [ ] Root Directory: src/backend
- [ ] Environment: Docker
- [ ] Health Check Path: /health
- [ ] Variables de entorno (4 variables)
- [ ] Create Web Service
- [ ] Esperar 5 minutos
- [ ] Copiar URL del backend
- [ ] Actualizar VITE_API_URL en frontend
- [ ] Esperar 2 minutos
- [ ] Verificar que funciona

---

**EMPIEZA AHORA:** https://dashboard.render.com 🚀

**Tiempo total: 10 minutos**
