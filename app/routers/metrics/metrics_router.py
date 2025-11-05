from fastapi import APIRouter, HTTPException
from pathlib import Path
import json
from typing import Dict, List
from datetime import datetime, timedelta
import hashlib

router = APIRouter(
    prefix="/metrics",
    tags=["metrics"]
)

CURRENT_DIR = Path(__file__).parent.parent
METRICS_FILE = CURRENT_DIR / "data" / "metrics.json"


def load_metrics() -> Dict:
    """Lädt Metriken aus JSON-Datei"""
    try:
        if not METRICS_FILE.exists():
            initial_data = {
                "version": "1.0",
                "total_requests": 0,
                "entries": []
            }
            save_metrics(initial_data)
            return initial_data
        
        with open(METRICS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"✗ Fehler beim Laden der Metriken: {e}")
        return {"version": "1.0", "total_requests": 0, "entries": []}


def save_metrics(data: Dict) -> bool:
    try:
        with open(METRICS_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"✗ Fehler beim Speichern der Metriken: {e}")
        return False


def calculate_image_hash(image_base64: str) -> str:
    """Berechnet Hash des Bildes für Duplikat-Erkennung"""
    return hashlib.md5(image_base64.encode()).hexdigest()[:16]


@router.post("/log")
async def log_metric(metric_data: Dict):
    try:
        metrics = load_metrics()
        
        entry = {
            "id": metrics["total_requests"] + 1,
            "timestamp": datetime.now().isoformat(),
            "success": metric_data.get("success", False),
            "recipient": metric_data.get("recipient", "Unbekannt"),
            "confidence": metric_data.get("confidence", 0),
            "matched_from_list": metric_data.get("matched_from_list", False),
            "method": metric_data.get("method", "unknown"),
            "ocr_strategy": metric_data.get("ocr_strategy", "unknown"),
            "ocr_engine": metric_data.get("ocr_engine", "unknown"),
            "ocr_quality": metric_data.get("ocr_quality", 0),
            "image_quality_score": metric_data.get("image_quality_score", 0),
            "processing_time": metric_data.get("processing_time", 0),
            "image_size": metric_data.get("image_size", 0),
            "image_format": metric_data.get("image_format", "Unknown"),
            "image_hash": metric_data.get("image_hash", ""),
            "word_pool": metric_data.get("word_pool", []),
            "error": metric_data.get("error", None)
        }
        
        metrics["entries"].append(entry)
        metrics["total_requests"] += 1

        save_metrics(metrics)
        
        return {
            "success": True,
            "message": "Metrik erfolgreich geloggt",
            "entry_id": entry["id"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fehler beim Loggen: {str(e)}")


@router.get("/stats")
async def get_statistics():
    try:
        metrics = load_metrics()
        entries = metrics.get("entries", [])
        
        if not entries:
            return {
                "total_requests": 0,
                "success_rate": 0,
                "avg_confidence": 0,
                "avg_processing_time": 0,
                "avg_image_quality": 0,
                "message": "Noch keine Daten vorhanden"
            }
        
        total = len(entries)
        successful = sum(1 for e in entries if e.get("matched_from_list", False))
        
        confidences = [e.get("confidence", 0) for e in entries if e.get("confidence", 0) > 0]
        processing_times = [e.get("processing_time", 0) for e in entries]
        image_qualities = [e.get("image_quality_score", 0) for e in entries if e.get("image_quality_score", 0) > 0]
        
        engine_counts = {}
        for e in entries:
            engine = e.get("ocr_engine", "unknown")
            engine_counts[engine] = engine_counts.get(engine, 0) + 1
        
        recipient_counts = {}
        for e in entries:
            if e.get("matched_from_list", False):
                recipient = e.get("recipient", "Unbekannt")
                recipient_counts[recipient] = recipient_counts.get(recipient, 0) + 1
        
        top_recipients = sorted(recipient_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        confidence_ranges = {
            "0-50": 0,
            "50-70": 0,
            "70-85": 0,
            "85-95": 0,
            "95-100": 0
        }
        
        for conf in confidences:
            if conf < 50:
                confidence_ranges["0-50"] += 1
            elif conf < 70:
                confidence_ranges["50-70"] += 1
            elif conf < 85:
                confidence_ranges["70-85"] += 1
            elif conf < 95:
                confidence_ranges["85-95"] += 1
            else:
                confidence_ranges["95-100"] += 1
        
        return {
            "total_requests": total,
            "successful_matches": successful,
            "failed_matches": total - successful,
            "success_rate": round((successful / total) * 100, 2),
            "avg_confidence": round(sum(confidences) / len(confidences), 2) if confidences else 0,
            "avg_processing_time": round(sum(processing_times) / len(processing_times), 2) if processing_times else 0,
            "avg_image_quality": round(sum(image_qualities) / len(image_qualities), 2) if image_qualities else 0,
            "ocr_engines": engine_counts,
            "top_recipients": [{"name": r[0], "count": r[1]} for r in top_recipients],
            "confidence_distribution": confidence_ranges
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fehler: {str(e)}")


@router.get("/recent")
async def get_recent_entries(limit: int = 20):
    try:
        metrics = load_metrics()
        entries = metrics.get("entries", [])
        
        recent = sorted(entries, key=lambda x: x.get("timestamp", ""), reverse=True)[:limit]
        
        return {
            "count": len(recent),
            "entries": recent
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fehler: {str(e)}")


@router.get("/timeline")
async def get_timeline(days: int = 7):
    try:
        metrics = load_metrics()
        entries = metrics.get("entries", [])
        
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_entries = [
            e for e in entries 
            if datetime.fromisoformat(e.get("timestamp", "")) > cutoff_date
        ]
        
        daily_stats = {}
        for entry in recent_entries:
            date = entry.get("timestamp", "")[:10]
            
            if date not in daily_stats:
                daily_stats[date] = {
                    "date": date,
                    "total": 0,
                    "successful": 0,
                    "failed": 0
                }
            
            daily_stats[date]["total"] += 1
            if entry.get("matched_from_list", False):
                daily_stats[date]["successful"] += 1
            else:
                daily_stats[date]["failed"] += 1
        
        timeline = sorted(daily_stats.values(), key=lambda x: x["date"])
        
        return {
            "days": days,
            "timeline": timeline
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fehler: {str(e)}")


@router.get("/export")
async def export_metrics():
    try:
        metrics = load_metrics()
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fehler: {str(e)}")


@router.delete("/clear")
async def clear_metrics():
    try:
        initial_data = {
            "version": "1.0",
            "total_requests": 0,
            "entries": []
        }
        save_metrics(initial_data)
        
        return {
            "success": True,
            "message": "Alle Metriken gelöscht"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fehler: {str(e)}")