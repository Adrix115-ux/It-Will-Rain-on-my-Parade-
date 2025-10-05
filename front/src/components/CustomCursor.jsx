// src/components/CustomCursor.jsx
import React, { useEffect, useRef } from "react";
import tatu from "../assets/tatuverde.png";
import "./CustomCursor.css";

export default function CustomCursor({ enabled = true, size = 24 }) {
    const imgRef = useRef(null);

    useEffect(() => {
        if (!enabled) return;
        const img = imgRef.current;
        if (!img) return;

        const onMove = (e) => {
            // offset to center the tiny image on the pointer
            const offset = Math.round(size / 2);
            img.style.transform = `translate3d(${e.clientX - offset}px, ${e.clientY - offset}px, 0)`;
        };
        const onEnter = () => {
            img.style.opacity = "1";
        };
        const onLeave = () => {
            img.style.opacity = "0";
        };

        document.addEventListener("mousemove", onMove);
        document.addEventListener("mouseenter", onEnter);
        document.addEventListener("mouseleave", onLeave);

        // hide native cursor
        document.documentElement.style.cursor = "none";

        return () => {
            document.removeEventListener("mousemove", onMove);
            document.removeEventListener("mouseenter", onEnter);
            document.removeEventListener("mouseleave", onLeave);
            document.documentElement.style.cursor = ""; // restore
        };
    }, [enabled, size]);

    return (
        <img
            ref={imgRef}
            src={tatu}
            alt=""
            aria-hidden="true"
            className="app-custom-cursor"
            style={{
                width: `${size}px`,
                height: `${size}px`,
                // slight scale / smoothing
                transform: `translate3d(-50%,-50%,0)`,
            }}
        />
    );
}

