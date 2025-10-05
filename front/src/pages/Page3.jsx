// ASCII only
import React, { useEffect, useMemo, useState } from "react";
import { useLocation } from "react-router-dom";
import api from "../utils/api";

const dateToYYYYMMDD = (d) => {
    const yyyy=d.getFullYear(), mm=String(d.getMonth()+1).padStart(2,"0"), dd=String(d.getDate()).padStart(2,"0");
    return `${yyyy}${mm}${dd}`;
};

const MISSING = new Set([-999,-8888,-99,-9]);
const miss = (v)=> v==null || Number.isNaN(v) || (typeof v==="number" && MISSING.has(Math.trunc(v)));

export default function Page3(){
    const { state } = useLocation();
    const [rows,setRows]=useState(Array.isArray(state?.rows)? state.rows: []);
    const [loading,setLoading]=useState(!rows.length);
    const [error,setError]=useState("");

    // if rows not provided, fetch with a safe parameter set
    useEffect(()=>{
        if(rows.length) return; // already provided by Page1
        let cancelled=false;
        (async ()=>{
            setLoading(true); setError("");
            try{
                const lat=-17.6, lon=-62.92;
                const today=new Date(); const start=new Date(today); start.setDate(today.getDate()-6);
                const startS=dateToYYYYMMDD(start);

                // Safe list: todo lo que suele existir en "daily"
                const params = [
                    "T2M","T2M_MAX","T2M_MIN","T2MDEW","RH2M",
                    "PRECTOTCORR","WS10M","WD10M","PS",
                    "ALLSKY_SFC_SW_DWN","CLRSKY_SFC_SW_DWN","ALLSKY_SFC_SW_DIFF",
                    "TOA_SW_DWN","ALBEDO","ALLSKY_SFC_PAR_TOT"
                ].join(",");

                const {data:payload}=await api.get("/ClimateLogic/weekly",{
                    params:{ latitude:lat, longitude:lon, start:startS, days:7, parametersCsv:params }
                });

                if(cancelled) return;
                const data = Array.isArray(payload?.data)? payload.data: [];
                const cleaned = data.map((r)=>{
                    const o={...r};
                    Object.keys(o).forEach(k=>{
                        const v=o[k]; if(typeof v==="number" && miss(v)) o[k]=null;
                    });
                    return o;
                });

                setRows(cleaned);
            }catch(e){
                console.error("page3 fetch failed:", e);
                setError("No se pudo cargar datos avanzados.");
            }finally{
                if(!cancelled) setLoading(false);
            }
        })();
        return ()=>{cancelled=true;};
    },[rows.length]);

    const columns = useMemo(()=>{
        const s=new Set(["date"]);
        rows.forEach(r=>Object.keys(r).forEach(k=>s.add(k)));
        const cols=Array.from(s);
        cols.sort((a,b)=> (a==="date"? -1 : b==="date"? 1 : a.localeCompare(b)));
        return cols;
    },[rows]);

    const downloadCsv=()=>{
        const header=columns.join(",");
        const lines=rows.map(r=>columns.map(c=>{
            const v=r[c]; if(v==null) return "";
            const s=String(v); return /[",\n]/.test(s)? `"${s.replace(/"/g,'""')}"` : s;
        }).join(","));
        const csv=[header,...lines].join("\n");
        const blob=new Blob([csv],{type:"text/csv;charset=utf-8"});
        const url=URL.createObjectURL(blob); const a=document.createElement("a");
        a.href=url; a.download="weekly_data.csv"; a.click(); URL.revokeObjectURL(url);
    };

    return (
        <div className="page-container centered">
            <h1>DATOS AVANZADOS (ULTIMA SEMANA)</h1>
            {loading && <p>Cargando...</p>}
            {error && <p style={{color:"red"}}>{error}</p>}

            <button className="btn" onClick={downloadCsv} disabled={!rows.length} style={{marginBottom:12}}>
                Descargar CSV
            </button>

            <div style={{overflowX:"auto"}}>
                <table className="table table-sm" style={{width:"100%", borderCollapse:"collapse"}}>
                    <thead>
                    <tr>
                        {columns.map(c=>(
                            <th key={c} style={{borderBottom:"1px solid #e5e7eb", padding:"6px 8px", textAlign:"left"}}>{c}</th>
                        ))}
                    </tr>
                    </thead>
                    <tbody>
                    {rows.map(r=>(
                        <tr key={r.date}>
                            {columns.map(c=>(
                                <td key={c} style={{borderBottom:"1px solid #f2f2f2", padding:"6px 8px"}}>
                                    {r[c]==null? "—" : r[c]}
                                </td>
                            ))}
                        </tr>
                    ))}
                    {!rows.length && !loading && (
                        <tr><td colSpan={columns.length} style={{padding:12}}>Sin datos</td></tr>
                    )}
                    </tbody>
                </table>
            </div>
        </div>
    );
}
