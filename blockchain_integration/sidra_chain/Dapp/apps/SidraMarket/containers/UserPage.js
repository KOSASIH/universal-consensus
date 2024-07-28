import React, { useState, useEffect } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { fetchUser } from '../actions/userActions';
import UserForm from '../components/UserForm';

const UserPage = () => {
  const [user, setUser] = useState({});
  const dispatch = useDispatch();
  const userId = useSelector((state) => state.auth.userId);

  useEffect(() => {
    const fetchUserData = async () => {
      try {
        const userData = await dispatch(fetchUser(userId));
        setUser(userData);
      } catch (error) {
        console.error(error);
      }
    };
    fetchUserData();
  }, [userId, dispatch]);

  const handleSubmit = async (values) => {
    try {
      await dispatch(updateUser(values));
    } catch (error) {
      console.error(error);
    }
  };

  return (
    <div>
      <h1>User Page</h1>
      <UserForm user={user} onSubmit={handleSubmit} />
    </div>
  );
};

export default UserPage;
