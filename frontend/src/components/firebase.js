import { initializeApp } from "firebase/app";
import { getFirestore } from "@firebase/firestore";

const firebaseConfig = {
  apiKey: "AIzaSyCVrwVHFwD2WfFfpmTcG_znJPocjf5Lh5Q",
  authDomain: "mumbaihacks-d2969.firebaseapp.com",
  projectId: "mumbaihacks-d2969",
  storageBucket: "mumbaihacks-d2969.appspot.com",
  messagingSenderId: "5380375469",
  appId: "1:5380375469:web:e47df3b0758444667b9724",
  measurementId: "G-Z3N5CBKVXS"
};

const app = initializeApp(firebaseConfig);
const db = getFirestore(app);

export { db };