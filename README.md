# Real-Time Multimodal Emergency Detection System

An end-to-end backend system designed to detect emergency situations in real time using audio and visual signals.  
The project focuses on reliability, real-time processing, and clean service boundaries.

## 🔍 Overview
This system processes live audio and visual inputs to identify emergency-related events, even when individuals may be unable to communicate clearly or in English. It combines machine learning models with backend services and containerised deployment.

## ✨ Key Features
- Real-time audio and image-based event detection
- REST APIs for inference and system interaction
- Containerised deployment using Docker
- Monitoring dashboard for system behaviour and outputs
- Modular design allowing independent evolution of components

## 🧠 Architecture
The system is designed as a modular, distributed pipeline that prioritises scalability, fault isolation and real-time performance. Components are decoupled to allow independent deployment, scaling and evolution.

Ingestion Layer: Audio streams and video frames are ingested as independent, asynchronous data flows. Pre-processing is handled close to ingestion to minimise downstream load and reduce end-to-end latency.

Stateless Inference Services: Audio and vision models run in isolated, stateless services and expose inference via REST APIs. This design supports horizontal scaling, rolling updates and independent performance tuning of each service.

Orchestration & Decision Layer: A coordinating service aggregates inference outputs, applies decision logic, and manages timing constraints to ensure consistent real-time responses across modalities.

Data Flow & Back-Pressure: Data is processed in a streaming manner with minimal shared state, reducing coupling and enabling graceful degradation under load. The architecture is designed to support back-pressure and async messaging as system demand grows.

Observability & Reliability: System outputs, health checks and performance metrics are surfaced through a monitoring dashboard to support debugging, failure detection and operational visibility.

Deployment & Reproducibility: All services are containerised using Docker, enabling reproducible builds, environment parity and a clear path to automated deployment and orchestration.
