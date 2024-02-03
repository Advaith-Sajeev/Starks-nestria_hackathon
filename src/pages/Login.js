import React from "react";
import "../styles/Login.css";
import VisibilityIcon from "@mui/icons-material/Visibility";
import EmailIcon from "@mui/icons-material/Email";
import axios from "axios";
import { useState } from "react";
import { useNavigate, Link } from "react-router-dom";

function Login() {
  const history=useNavigate();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");

 async function submit(e) {
  e.preventDefault();
  if(!username||!password){
  alert("Please fill all the fields");
  return;
  }
  try {
    const response = await fetch(`http://127.0.0.1:8000/login?username=${username}&password=${password}`);
    if(response.status === 200){
    history("/home",{state:{username:username}});
    }
    else if (response.status === 400){
    alert("Incorrect Credentials");
    }
    console.log(response.data);
  } catch (error) {
    console.error("Error during signup:", error);
  }
}

  return (
    <div className="login">
      <div className="header">
        <div className="text">Login</div>
        <div className="undeline"></div>
      </div>
      <form action="POST">
        <div className="inputs">
          <div className="input">
            <EmailIcon className="img" />
            <input
              onChange={(e) => {
                setUsername(e.target.value);
              }}
              placeholder="Username"
            ></input>
          </div>
          <div className="input">
            <VisibilityIcon className="img" />
            <input
              type="password"
              onChange={(e) => {
                setPassword(e.target.value);
              }}
              placeholder="Password"
            ></input>
          </div>
        </div>
      </form>

      <div className="submit-container">
        <div className="submit-bttn">
          <input type="submit" className="submit" onClick={submit}></input>
        </div>
        <Link to="/register" className="link">
          <div className="submit">Sign Up</div>
        </Link>
      </div>
    </div>
  );
}

export default Login;
