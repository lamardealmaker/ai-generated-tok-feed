import 'package:firebase_auth/firebase_auth.dart';
import 'package:get/get.dart';
import '../models/user_model.dart';
import '../services/user_service.dart';

class AuthController extends GetxController {
  static AuthController instance = Get.find();
  
  // Observables
  Rx<User?> _user = Rx<User?>(null);
  Rx<UserModel?> _userModel = Rx<UserModel?>(null);
  RxBool isLoading = false.obs;

  // Getters
  User? get user => _user.value;
  UserModel? get userModel => _userModel.value;
  bool get isLoggedIn => _user.value != null;
  bool get isEmailVerified => _user.value?.emailVerified ?? false;

  // Firebase instances
  final FirebaseAuth _auth = FirebaseAuth.instance;
  final UserService _userService = UserService();

  @override
  void onReady() {
    super.onReady();
    _user = Rx<User?>(_auth.currentUser);
    _user.bindStream(_auth.authStateChanges());
    ever(_user, _setInitialScreen);
  }

  void _setInitialScreen(User? user) async {
    if (user == null) {
      _userModel.value = null;
      Get.offAllNamed('/login');
    } else {
      await _fetchUserModel(user.uid);
      
      if (!user.emailVerified) {
        Get.offAllNamed('/verify-email');
      } else {
        Get.offAllNamed('/home');
      }
    }
  }

  Future<void> _fetchUserModel(String uid) async {
    try {
      UserModel? user = await _userService.getUserById(uid);
      if (user != null) {
        _userModel.value = user;
      }
    } catch (e) {
      print('Error fetching user model: $e');
    }
  }

  // Sign up with email and password
  Future<String> signUp({
    required String email,
    required String password,
    String? username,
    String? fullName,
  }) async {
    try {
      isLoading.value = true;
      
      // Create user with email and password
      UserCredential cred = await _auth.createUserWithEmailAndPassword(
        email: email,
        password: password,
      );

      // Update display name if username is provided
      if (username != null) {
        await cred.user!.updateDisplayName(username);
      }

      // Send email verification
      await cred.user!.sendEmailVerification();

      // Create user model
      UserModel newUser = UserModel(
        uid: cred.user!.uid,
        email: email,
        username: username,
        fullName: fullName,
        isEmailVerified: false,
        createdAt: DateTime.now(),
        lastLogin: DateTime.now(),
        userType: 'buyer',
      );

      // Create user in Firestore
      await _userService.createUser(newUser);
      
      _userModel.value = newUser;
      return 'success';
    } on FirebaseAuthException catch (e) {
      return _handleAuthError(e);
    } catch (e) {
      return e.toString();
    } finally {
      isLoading.value = false;
    }
  }

  // Login with email and password
  Future<String> login({
    required String email,
    required String password,
  }) async {
    try {
      isLoading.value = true;
      
      UserCredential cred = await _auth.signInWithEmailAndPassword(
        email: email,
        password: password,
      );

      // Update last login
      await _userService.updateLastLogin(cred.user!.uid);
      await _fetchUserModel(cred.user!.uid);
      
      return 'success';
    } on FirebaseAuthException catch (e) {
      return _handleAuthError(e);
    } catch (e) {
      return e.toString();
    } finally {
      isLoading.value = false;
    }
  }

  // Send email verification
  Future<String> sendEmailVerification() async {
    try {
      await _auth.currentUser?.sendEmailVerification();
      return 'success';
    } catch (e) {
      return e.toString();
    }
  }

  // Check email verification status
  Future<bool> checkEmailVerified() async {
    try {
      await _auth.currentUser?.reload();
      final isVerified = _auth.currentUser?.emailVerified ?? false;
      if (isVerified && _userModel.value != null) {
        await _userService.updateUser(
          _userModel.value!.uid,
          {'isEmailVerified': true}
        );
      }
      return isVerified;
    } catch (e) {
      return false;
    }
  }

  // Sign out
  Future<void> signOut() async {
    await _auth.signOut();
    _userModel.value = null;
  }

  // Password reset
  Future<String> resetPassword(String email) async {
    try {
      await _auth.sendPasswordResetEmail(email: email);
      return 'success';
    } on FirebaseAuthException catch (e) {
      return _handleAuthError(e);
    } catch (e) {
      return e.toString();
    }
  }

  // Refresh user model
  Future<void> refreshUserModel() async {
    if (_user.value != null) {
      await _fetchUserModel(_user.value!.uid);
    }
  }

  String _handleAuthError(FirebaseAuthException e) {
    switch (e.code) {
      case 'user-not-found':
        return 'No user found with this email.';
      case 'wrong-password':
        return 'Wrong password provided.';
      case 'email-already-in-use':
        return 'Email is already in use.';
      case 'invalid-email':
        return 'Invalid email address.';
      case 'weak-password':
        return 'Password is too weak.';
      case 'operation-not-allowed':
        return 'Operation not allowed.';
      case 'user-disabled':
        return 'User has been disabled.';
      case 'too-many-requests':
        return 'Too many attempts. Please try again later.';
      default:
        return e.message ?? 'An error occurred.';
    }
  }
}
