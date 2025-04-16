import React from "react";
import { Routes, Route } from "react-router-dom";
import Navbar from "./pages/Navbar/navbar";
import Trading from "./pages/Trading/trading";
import './App.css';

function App() {
  return (
    <>
      <Navbar />
      <Routes>
        <Route path="/" element={<Trading />} />
      </Routes>
    </>
  );
}

export default App;
