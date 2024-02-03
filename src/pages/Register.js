import "../styles/Register.css";
import PersonIcon from "@mui/icons-material/Person";
import VisibilityIcon from "@mui/icons-material/Visibility";
import EmailIcon from "@mui/icons-material/Email";
import { useState } from "react";
import { useNavigate, Link } from "react-router-dom";

function Register() {

  const history=useNavigate();
  const [name, setName] = useState("");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");

  async function submit(e) {
    e.preventDefault();
    try {
        const res= await fetch(`http://127.0.0.1:8000/signup?name=${name}&username=${username}&password=${password}`);
        console.log(res)
        if(res.status === 200){
    history("/login");
    }
    else if(res.status === 400){
    alert("Username already exists");
    }

    } catch (error) {
        window.alert("Error")
    }
}

  return (
    <div className="login">
      <div className="header">
        <div className="text">Sign Up</div>
        <div className="undeline"></div>
      </div>
      <form action="POST">
        <div className="inputs">
          <div className="input">
            <PersonIcon className="img" />
            <input
              onChange={(e) => {
                setName(e.target.value);
              }}
              placeholder="Name"
            ></input>
          </div>
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
        <Link to="/login" className="link">
          <div className="submit">Login</div>
        </Link>
      </div>
    </div>
  );
}

export default Register;
